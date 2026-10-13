import numpy as np

from mmaction.registry import TRANSFORMS

# Per-modality (mean, std), applied after FormatShape.
# RGB frames are first scaled to [0, 1] (ImageNet statistics); event frames
# are left untouched (raw [0, 255] values, mean 0 / std 1).
MEAN_STD = {
    'RGB': (np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])),
    'Event': (np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])),
}


@TRANSFORMS.register_module()
class NormalizePerModality:
    """Normalize ``results['imgs']`` according to ``results['modality']``.

    Expects channel-first frames (i.e. placed after ``FormatShape``) and
    outputs a stacked float32 array.
    """

    def __init__(self, to_float32=True):
        self.to_float32 = to_float32

    def __call__(self, results):
        imgs = results['imgs']
        modality = results['modality']
        mean, std = MEAN_STD.get(modality, MEAN_STD['RGB'])

        # Broadcast over (C, H, W) or (C, T, H, W).
        extra_dims = (1, ) * (imgs[0].ndim - 1)
        mean = mean.reshape((3, ) + extra_dims)
        std = std.reshape((3, ) + extra_dims)

        proc = []
        for img in imgs:
            if self.to_float32 and img.dtype != np.float32:
                img = img.astype(np.float32)
            if modality != 'Event':
                img = img / 255.0
            img = (img - mean) / std
            proc.append(img.astype(np.float32))

        results['imgs'] = np.stack(proc, axis=0)
        return results
