import copy
import os.path as osp

import numpy as np
from mmcv.transforms import Compose

from mmaction.datasets import RawframeDataset
from mmaction.registry import DATASETS, TRANSFORMS

from .hardvs_paths import (hardvs_frame_info, read_ann_csv, remap_frame_inds,
                           sequence_dir_from_ann_path)


@DATASETS.register_module()
class HARDVSPairedDataset(RawframeDataset):
    """Paired HARDVS APS (RGB) / DVS (event) dataset for cross-modal KD.

    Each sample holds the event frames for the student (``inputs``) and the
    RGB frames of the same recording for the teacher (``inputs_teacher``).

    Frames are sampled once on the event branch (``pipeline_student``); the
    same temporal positions are reused, remapped proportionally, on the RGB
    branch (``pipeline_teacher``). With ``sync_spatial_aug=True`` the
    student's random crop / flip are replayed on the teacher branch so both
    views cover the same region (needed by spatial feature KD such as
    attention transfer).

    Args:
        ann_file (str): APS annotation CSV (teacher / RGB).
        ann_file_event (str): DVS annotation CSV (student / event). Samples
            follow its order, so a run without KD sees the same sample order
            as the event-only baseline.
        pipeline_student (list[dict]): Event pipeline. Must start with a
            frame sampling transform.
        pipeline_teacher (list[dict]): RGB pipeline. Its sampling transform
            only runs to keep the pipeline structure; its frame indices are
            replaced by the remapped student indices.
        load_teacher (bool): If False, only the student branch is loaded.
        sync_spatial_aug (bool): Replay the student's crop / flip on the
            teacher branch.
    """

    def __init__(self,
                 ann_file,
                 ann_file_event,
                 pipeline_student,
                 pipeline_teacher,
                 load_teacher=True,
                 sync_spatial_aug=False,
                 data_prefix=dict(img=''),
                 test_mode=False,
                 **kwargs):
        self.ann_file_event = ann_file_event
        self.load_teacher = load_teacher
        self.sync_spatial_aug = sync_spatial_aug
        kwargs.pop('pipeline', None)

        # Both pipelines are applied manually in prepare_data.
        super().__init__(
            ann_file=ann_file,
            pipeline=[],
            data_prefix=data_prefix,
            test_mode=test_mode,
            **kwargs)

        self.pipeline_student = Compose(
            [TRANSFORMS.build(p) for p in pipeline_student])
        self.pipeline_teacher = Compose(
            [TRANSFORMS.build(p) for p in pipeline_teacher])

    @staticmethod
    def _read_by_sequence(ann_file):
        return {
            sequence_dir_from_ann_path(rel_path): label
            for rel_path, label in read_ann_csv(ann_file)
        }

    def load_data_list(self):
        """Pair APS and DVS rows sharing the same recording directory."""
        rgb_samples = self._read_by_sequence(self.ann_file)
        event_samples = self._read_by_sequence(self.ann_file_event)
        data_root = self.data_prefix.get('img') or osp.dirname(self.ann_file)

        data_list = []
        missing = 0
        label_mismatches = 0
        common_paths = rgb_samples.keys() & event_samples.keys()

        for rel_path in event_samples.keys():
            if rel_path not in common_paths:
                continue
            rgb_label = rgb_samples[rel_path]
            if rgb_label != event_samples[rel_path]:
                label_mismatches += 1
                continue
            if self.num_classes is not None and not (
                    0 <= rgb_label < self.num_classes):
                continue

            sequence_dir = osp.join(data_root, rel_path)
            try:
                rgb_info = hardvs_frame_info(sequence_dir, 'aps', 'RGB')
                event_info = hardvs_frame_info(sequence_dir, 'dvs', 'Event')
            except FileNotFoundError:
                missing += 1
                continue

            data_list.append(
                dict(
                    seq_id=rel_path,
                    label=rgb_label,
                    rgb_info=rgb_info,
                    event_info=event_info,
                ))

        if not data_list:
            raise RuntimeError(
                f'No valid HARDVS RGB/Event pairs. data_root={data_root!r}, '
                f'APS annotations={len(rgb_samples)}, '
                f'DVS annotations={len(event_samples)}, '
                f'common_paths={len(common_paths)}, missing={missing}, '
                f'label_mismatches={label_mismatches}.')

        print(f'Paired {len(data_list)} HARDVS APS+DVS sequences '
              f'(missing={missing}, label_mismatches={label_mismatches})')
        return data_list

    # ------------------------------------------------------------------
    # Pipeline helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _transform_name(transform):
        return transform.__class__.__name__

    def _apply_until_sampleframes(self, data, pipeline, branch):
        """Run a pipeline until frame indices have been sampled."""
        results = copy.deepcopy(data)

        for idx, transform in enumerate(pipeline.transforms):
            transform_name = self._transform_name(transform)
            try:
                results = transform(results)
            except Exception as exc:
                raise RuntimeError(
                    f'{branch} pipeline failed in {transform_name} for '
                    f'{data.get("frame_dir")!r}: {exc}') from exc
            if results is None:
                raise RuntimeError(
                    f'{branch} pipeline transform {transform_name} returned '
                    f'None for {data.get("frame_dir")!r}')
            if 'frame_inds' in results:
                return results, idx + 1

        raise RuntimeError(
            f'{branch} pipeline did not produce frame_inds for '
            f'{data.get("frame_dir")!r}. Check that it contains a frame '
            f'sampling transform.')

    def _apply_from_index(self, data, pipeline, start_idx, branch,
                          record=None, replay=None):
        """Run the remainder of a pipeline from a given transform index.

        If ``record`` is a list, the outcome of each random spatial transform
        (crop / flip) is appended to it. If ``replay`` is a list, random
        spatial transforms are not drawn but replayed from it, in order.
        """
        results = copy.deepcopy(data)
        if replay is not None:
            replay = list(replay)

        for transform in pipeline.transforms[start_idx:]:
            transform_name = self._transform_name(transform)
            kind = self._sync_kind(transform)
            try:
                if kind is not None and replay is not None:
                    results = self._replay_spatial(results, transform, kind,
                                                   replay)
                elif kind is not None and record is not None:
                    results = self._record_spatial(results, transform, kind,
                                                   record)
                else:
                    results = transform(results)
            except Exception as exc:
                raise RuntimeError(
                    f'{branch} pipeline failed in {transform_name} for '
                    f'{data.get("frame_dir")!r}: {exc}') from exc
            if results is None:
                raise RuntimeError(
                    f'{branch} pipeline transform {transform_name} returned '
                    f'None for {data.get("frame_dir")!r}')

        if replay and any(value is not None for _, value in replay):
            raise RuntimeError(
                f'{branch} pipeline has fewer random crop/flip transforms '
                f'than the student pipeline; cannot keep views aligned.')

        return results

    _SYNC_CROP_TYPES = ('MultiScaleCrop', 'RandomCrop', 'RandomResizedCrop')
    _SYNC_FLIP_TYPES = ('Flip', )

    @staticmethod
    def _unwrap(transform):
        # Wrapper transforms keep the real transform in `op`.
        return getattr(transform, 'op', transform)

    def _sync_kind(self, transform):
        name = self._transform_name(self._unwrap(transform))
        if name in self._SYNC_CROP_TYPES:
            return 'crop'
        if name in self._SYNC_FLIP_TYPES:
            return 'flip'
        return None

    @staticmethod
    def _record_spatial(results, transform, kind, record):
        """Apply a random spatial transform and store its outcome.

        The crop is stored relative to the pre-crop image size so it can be
        replayed on a teacher frame of a different resolution. ``None`` is
        stored when the transform was not applied.
        """
        if results.get('lazy'):
            raise NotImplementedError(
                'sync_spatial_aug does not support lazy transforms')

        if kind == 'crop':
            img_h, img_w = results['img_shape']
            prev_bbox = results.get('crop_bbox')
            results = transform(results)
            bbox = results.get('crop_bbox')
            applied = bbox is not None and (
                prev_bbox is None or not np.array_equal(bbox, prev_bbox))
            record.append((kind, np.asarray(bbox, dtype=np.float64) /
                           np.array([img_w, img_h, img_w, img_h])
                           if applied else None))
        else:
            had_flip = 'flip' in results
            results = transform(results)
            applied = 'flip' in results and not had_flip
            record.append((kind, (bool(results['flip']),
                                  results['flip_direction'])
                           if applied else None))
        return results

    def _replay_spatial(self, results, transform, kind, replay):
        """Apply the student's recorded crop/flip instead of drawing one."""
        if not replay:
            raise RuntimeError(
                'Teacher pipeline has more random crop/flip transforms than '
                'the student pipeline; cannot keep views aligned.')
        rec_kind, value = replay.pop(0)
        if rec_kind != kind:
            raise RuntimeError(
                f'Student/teacher random spatial transforms are in a '
                f'different order (student: {rec_kind}, teacher: {kind}).')
        if value is None:
            return results
        if results.get('lazy'):
            raise NotImplementedError(
                'sync_spatial_aug does not support lazy transforms')

        if kind == 'flip':
            flip, direction = value
            op = copy.copy(self._unwrap(transform))
            op.flip_ratio = 1.0 if flip else 0.0
            op.direction = direction
            return op(results)

        img_h, img_w = results['img_shape']
        x1, y1, x2, y2 = np.round(
            value * np.array([img_w, img_h, img_w, img_h])).astype(int)
        x2, y2 = min(x2, img_w), min(y2, img_h)
        results['imgs'] = [img[y1:y2, x1:x2] for img in results['imgs']]
        results['crop_bbox'] = np.array([x1, y1, x2, y2])
        results['img_shape'] = (y2 - y1, x2 - x1)

        # Same crop_quadruple bookkeeping as mmaction's RandomCrop family.
        old_x, old_y, old_w, old_h = results.get(
            'crop_quadruple', np.array([0, 0, 1, 1], dtype=np.float32))
        results['crop_quadruple'] = np.array([
            old_x + x1 / img_w * old_w, old_y + y1 / img_h * old_h,
            (x2 - x1) / img_w * old_w, (y2 - y1) / img_h * old_h
        ], dtype=np.float32)
        return results

    # ------------------------------------------------------------------
    # Sample preparation
    # ------------------------------------------------------------------
    @staticmethod
    def _branch_input(info, label, modality):
        return dict(
            frame_dir=info['frame_dir'],
            total_frames=info['total_frames'],
            frame_list=info['frame_list'],
            filename_tmpl=info['filename_tmpl'],
            label=label,
            start_index=1,
            modality=modality)

    def prepare_data(self, idx):
        data_info = self.get_data_info(idx)
        label = data_info['label']
        rgb_data = self._branch_input(data_info['rgb_info'], label, 'RGB')
        event_data = self._branch_input(data_info['event_info'], label,
                                        'Event')

        # Sample frames once on the student (event) branch.
        event_sampled, student_start_idx = self._apply_until_sampleframes(
            event_data, self.pipeline_student, 'student/Event')

        if not self.load_teacher:
            return self._apply_from_index(event_sampled,
                                          self.pipeline_student,
                                          student_start_idx, 'student/Event')

        rgb_sampled, teacher_start_idx = self._apply_until_sampleframes(
            rgb_data, self.pipeline_teacher, 'teacher/RGB')

        # Reuse the student's temporal sampling on the teacher branch.
        sample_keys = ('clip_len', 'frame_interval', 'num_clips', 'offset',
                       'twice_sample', 'out_of_bound_opt', 'avg_interval')
        for key in sample_keys:
            if key in event_sampled:
                rgb_sampled[key] = copy.deepcopy(event_sampled[key])
        rgb_sampled['frame_inds'] = remap_frame_inds(
            event_sampled['frame_inds'],
            data_info['event_info']['total_frames'],
            data_info['rgb_info']['total_frames'])

        spatial_record = [] if self.sync_spatial_aug else None
        event_result = self._apply_from_index(
            event_sampled, self.pipeline_student, student_start_idx,
            'student/Event', record=spatial_record)
        rgb_result = self._apply_from_index(
            rgb_sampled, self.pipeline_teacher, teacher_start_idx,
            'teacher/RGB', replay=spatial_record)

        event_result['inputs_teacher'] = rgb_result['inputs']
        return event_result

    def __getitem__(self, idx):
        result = self.prepare_data(idx)
        if result is None:
            raise RuntimeError(
                f'prepare_data returned None for index={idx}, '
                f'seq_id={self.get_data_info(idx).get("seq_id")!r}')
        return result
