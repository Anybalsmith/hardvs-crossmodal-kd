import cv2
import numpy as np
from mmaction.registry import TRANSFORMS


def gray_world(img):
    """Gray-world white balance: bring the three channels to the same mean. img: uint8 RGB HxWx3."""
    m = img.reshape(-1, 3).mean(0)
    return np.clip(img * (m.mean() / (m + 1e-6)), 0, 255).astype(np.uint8)


def clahe_l(img, clahe):
    """CLAHE on the L channel of LAB only, so colours are not distorted. img: uint8 RGB."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lab[..., 0] = clahe.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


@TRANSFORMS.register_module()
class ContrastEnhance:
    """Gray-world white balance then CLAHE on luminance, frame by frame.

    Put it right after HARDVSFrameDecode (uint8 RGB frames at native
    resolution), in the train, val and test pipelines. Frames whose modality
    is not 'RGB' (events, depth...) are left untouched.
    """

    def __init__(self, white_balance=True, clip_limit=2.0, tile_grid=4):
        self.white_balance = white_balance
        self.clip_limit = clip_limit
        self.tile_grid = tile_grid

    def __call__(self, results):
        if results.get('modality', 'RGB') != 'RGB':
            return results
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=(self.tile_grid, self.tile_grid))
        out = []
        for img in results['imgs']:
            assert img.dtype == np.uint8 and img.ndim == 3 and img.shape[2] == 3, \
                f'ContrastEnhance expects uint8 RGB frames, got {img.dtype} {img.shape}'
            if self.white_balance:
                img = gray_world(img)
            out.append(clahe_l(img, clahe))
        results['imgs'] = out
        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}(white_balance={self.white_balance}, '
                f'clip_limit={self.clip_limit}, tile_grid={self.tile_grid})')
