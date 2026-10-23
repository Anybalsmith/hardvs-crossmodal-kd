import os.path as osp

import cv2
import numpy as np
from PIL import Image
from mmaction.registry import TRANSFORMS


@TRANSFORMS.register_module()
class PersonMaskBackground:
    """Black out the background of RGB frames with a precomputed clip-level person mask.

    The mask of a clip is one uint8 PNG (0/255, frame size) at
    ``mask_root/<frame_dir relative to data_root>.png``, e.g. the union over
    the clip of the YOLO person masks, dilated
    (tools/data/hardvs/make_clip_masks.py). Its edge is
    softened by a Gaussian of size 2*feather+1, then each frame is multiplied
    by it. A clip whose mask is empty (no person found) is left unchanged.

    Put it right after HARDVSFrameDecode (uint8 RGB frames at native
    resolution), in the train, val and test pipelines. Frames whose modality
    is not 'RGB' are left untouched.
    """

    def __init__(self, mask_root, data_root, feather=7):
        self.mask_root = mask_root
        self.data_root = data_root
        self.feather = feather

    def __call__(self, results):
        if results.get('modality', 'RGB') != 'RGB':
            return results
        rel = osp.relpath(osp.normpath(results['frame_dir']), self.data_root)
        path = osp.join(self.mask_root, rel + '.png')
        if not osp.exists(path):
            raise FileNotFoundError(f'PersonMaskBackground: no clip mask {path}')
        mask = np.asarray(Image.open(path)) > 0
        if not mask.any():
            return results
        k = 2 * self.feather + 1
        alpha = cv2.GaussianBlur(mask.astype(np.float32), (k, k), 0)[..., None]
        out = []
        for img in results['imgs']:
            assert img.dtype == np.uint8 and img.shape[:2] == mask.shape, \
                f'PersonMaskBackground expects uint8 frames of the mask size {mask.shape}, got {img.dtype} {img.shape}'
            out.append((alpha * img).astype(np.uint8))
        results['imgs'] = out
        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}(mask_root={self.mask_root!r}, '
                f'data_root={self.data_root!r}, feather={self.feather})')
