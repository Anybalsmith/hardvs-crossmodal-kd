import copy as cp
import os.path as osp

import mmcv
import numpy as np
from mmengine.fileio import get as mm_get

from mmaction.datasets.transforms import RawFrameDecode
from mmaction.registry import TRANSFORMS


def infer_modality_from_path(frame_dir: str) -> str:
    """HARDVS ``<seq>_dvs`` folders hold event frames, anything else RGB."""
    if osp.basename(osp.normpath(frame_dir)).lower().endswith('_dvs'):
        return 'Event'
    return 'RGB'


def _decode_one(path: str, modality: str) -> np.ndarray:
    """Decode a single frame file. Raises on any failure (missing, empty or
    corrupt file), including errors raised inside mmcv/cv2."""
    data_bytes = mm_get(path)
    if data_bytes is None or len(data_bytes) == 0:
        raise RuntimeError(f'Empty bytes ({modality}) -> {path}')

    if modality == 'RGB':
        img = mmcv.imfrombytes(data_bytes, channel_order='rgb')
        if img is None or img.size == 0:
            raise RuntimeError(f'OpenCV failed (RGB) -> {path}')
        return img

    if modality == 'Event':
        arr = mmcv.imfrombytes(
            data_bytes, flag='unchanged', channel_order='rgb')
        if arr is None or arr.size == 0:
            raise RuntimeError(f'OpenCV failed (Event image) -> {path}')
        if arr.ndim == 2:  # H,W -> H,W,1
            arr = arr[..., None]
        if arr.shape[2] == 1:  # replicate to 3 channels
            arr = np.repeat(arr, 3, axis=-1)
        return arr.astype(np.float32)

    raise NotImplementedError(f'Unsupported modality {modality}')


def _frame_path(directory, idx, frame_list, filename_tmpl):
    if frame_list is not None:
        if not (1 <= idx <= len(frame_list)):
            return None
        return osp.join(directory, frame_list[idx - 1])
    return osp.join(directory, filename_tmpl.format(idx))


_warned_bad_frames = set()

# How many neighbouring indices to try (nearest first) when a frame fails to
# decode, before giving up on the whole sequence.
_MAX_DECODE_FALLBACK_RADIUS = 50


@TRANSFORMS.register_module()
class HARDVSFrameDecode(RawFrameDecode):
    """Decode HARDVS RGB (APS) or event (DVS) frames.

    The modality is inferred from ``frame_dir``. Event frames are kept as
    raw float32 values in [0, 255] (they are not divided by 255, see
    ``NormalizePerModality``). A missing or corrupt frame is replaced by the
    nearest readable frame of the same sequence instead of crashing the
    DataLoader worker.
    """

    def _decode_with_fallback(self, directory, idx, frame_list,
                              filename_tmpl, modality):
        path = _frame_path(directory, idx, frame_list, filename_tmpl)
        if path is not None and osp.exists(path):
            try:
                return _decode_one(path, modality)
            except Exception as exc:  # noqa: BLE001 - corrupt file
                first_error, bad_path = exc, path
        else:
            first_error = FileNotFoundError(f'Missing file -> {path}')
            bad_path = path

        if bad_path not in _warned_bad_frames:
            _warned_bad_frames.add(bad_path)
            print(f'[HARDVSFrameDecode] Failed to decode {bad_path!r} '
                  f'({first_error}) -- substituting nearest readable frame '
                  f'in {directory!r}.')

        total = len(frame_list) if frame_list is not None else None
        for delta in range(1, _MAX_DECODE_FALLBACK_RADIUS + 1):
            for cand in (idx - delta, idx + delta):
                if cand < 1 or (total is not None and cand > total):
                    continue
                cand_path = _frame_path(directory, cand, frame_list,
                                        filename_tmpl)
                if cand_path is None or not osp.exists(cand_path):
                    continue
                try:
                    return _decode_one(cand_path, modality)
                except Exception:  # noqa: BLE001
                    continue

        raise RuntimeError(
            f'[HARDVSFrameDecode] No readable frame found near index {idx} '
            f'in {directory!r} (modality={modality}).') from first_error

    def transform(self, results):
        mmcv.use_backend(self.decoding_backend)

        directory = results['frame_dir']
        modality = infer_modality_from_path(directory)

        frame_inds = np.squeeze(results['frame_inds'])
        offset = results.get('offset', 0)
        frame_list = results.get('frame_list')
        filename_tmpl = results.get('filename_tmpl')

        imgs, cache = [], {}
        for frame_idx in frame_inds:
            if frame_idx in cache:
                imgs.append(cp.deepcopy(imgs[cache[frame_idx]]))
                continue
            img = self._decode_with_fallback(directory, frame_idx + offset,
                                             frame_list, filename_tmpl,
                                             modality)
            cache[frame_idx] = len(imgs)
            imgs.append(img)

        results['imgs'] = imgs
        results['original_shape'] = imgs[0].shape[:2]
        results['img_shape'] = imgs[0].shape[:2]
        results['modality'] = modality
        return results
