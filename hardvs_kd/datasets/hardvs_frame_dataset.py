import os.path as osp

from mmaction.datasets import RawframeDataset
from mmaction.registry import DATASETS

from .hardvs_paths import (hardvs_frame_info, read_ann_csv,
                           sequence_dir_from_ann_path)


@DATASETS.register_module()
class HARDVSFrameDataset(RawframeDataset):
    """Single-modality HARDVS dataset, used by the RGB and Event baselines.

    ``ann_file`` is a ``path,label`` CSV whose paths end with ``<seq>_aps``
    (RGB) or ``<seq>_dvs`` (event); paths are relative to
    ``data_prefix['img']``. Recordings whose frame folder is missing or empty
    are skipped.
    """

    def load_data_list(self):
        data_root = self.data_prefix.get('img') or ''
        data_list = []
        missing = 0
        for rel_path, label in read_ann_csv(self.ann_file):
            last_part = rel_path.rstrip('/').rsplit('/', 1)[-1].lower()
            if last_part.endswith('_dvs'):
                suffix, modality = 'dvs', 'Event'
            elif last_part.endswith('_aps'):
                suffix, modality = 'aps', 'RGB'
            else:
                raise ValueError(
                    f'{self.ann_file}: expected a path ending with _aps or '
                    f'_dvs, got {rel_path!r}')

            sequence_dir = osp.join(data_root,
                                    sequence_dir_from_ann_path(rel_path))
            try:
                info = hardvs_frame_info(sequence_dir, suffix, modality)
            except FileNotFoundError:
                missing += 1
                continue
            info['label'] = label
            data_list.append(info)

        if not data_list:
            raise RuntimeError(
                f'No valid samples loaded from {self.ann_file} '
                f'(data_root={data_root!r}, missing={missing}).')
        if missing:
            print(f'[HARDVSFrameDataset] {self.ann_file}: skipped {missing} '
                  f'recordings with missing or empty frame folders.')
        return data_list
