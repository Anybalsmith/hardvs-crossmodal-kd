import numpy as np

from mmaction.datasets.transforms import SampleFrames
from mmaction.registry import TRANSFORMS


@TRANSFORMS.register_module()
class CadenceAugSampleFrames(SampleFrames):
    """``SampleFrames`` that sometimes turns a long recording into a short one.

    In HARDVS the number of APS frames per recording varies a lot with the
    session (from 2-4 PNGs, frames then repeated over the 8 segments, to
    40+ PNGs). When that number is correlated with the class in the training
    set, the model learns "repeated frames -> class X" instead of the gesture
    (HRI v4 subset: every training clip with <= 3 PNGs is A129).

    With probability ``prob`` (training only), a recording with more than
    ``max(num_distinct)`` frames is reduced to ``k`` frames, ``k`` drawn from
    ``num_distinct``: one random frame in each of ``k`` equal parts of the
    recording. The 8 segments are then sampled from these ``k`` frames with
    the parent ``SampleFrames`` logic, exactly as for a real ``k``-frame
    recording. Otherwise it behaves as ``SampleFrames``.

    ``force_test=True`` applies the reduction to every recording in test mode
    as well, deterministically (centre frame of each part, ``k`` =
    ``max(num_distinct)``); it is meant for diagnostics, not for evaluation.
    """

    def __init__(self, *args, prob=0.5, num_distinct=(3, 4),
                 force_test=False, **kwargs):
        super().__init__(*args, **kwargs)
        assert 0.0 <= prob <= 1.0
        self.prob = prob
        self.num_distinct = tuple(num_distinct)
        self.force_test = force_test

    def _pick_frames(self, total, k):
        edges = np.linspace(0, total, k + 1)
        if self.test_mode:
            return ((edges[:-1] + edges[1:]) / 2).astype(np.int64)
        lo = np.floor(edges[:-1]).astype(np.int64)
        hi = np.maximum(np.floor(edges[1:]).astype(np.int64), lo + 1)
        return np.random.randint(lo, hi)

    def transform(self, results):
        total = results['total_frames']
        if self.test_mode:
            apply = self.force_test
            k = max(self.num_distinct)
        else:
            apply = np.random.rand() < self.prob
            k = int(np.random.choice(self.num_distinct))
        if not apply or total <= max(self.num_distinct):
            return super().transform(results)

        chosen = self._pick_frames(total, k)
        start_index = results['start_index']
        results['total_frames'] = k
        results = super().transform(results)
        virtual = (results['frame_inds'] - start_index) % k
        results['frame_inds'] = (chosen[virtual] + start_index).astype(np.int32)
        results['total_frames'] = total
        results['cadence_frames'] = k
        return results

    def __repr__(self):
        return (f'{super().__repr__()[:-1]}, prob={self.prob}, '
                f'num_distinct={self.num_distinct}, '
                f'force_test={self.force_test})')
