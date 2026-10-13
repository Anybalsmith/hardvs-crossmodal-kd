"""Event-density aware dynamic sampling (EDS) for event/DVS frame sequences.

Inspired by section 4.2 ("Event-Density Aware Dynamic Sampling") of Wu et
al., "Towards Video Anomaly Detection from Event Streams: A Baseline and
Benchmark Datasets". Per-frame densities are read from a cache written by
``tools/data/hardvs/precompute_event_density.py``.
"""
import os.path as osp
from typing import Optional, Sequence, Tuple

import numpy as np
from mmcv.transforms import BaseTransform
from PIL import Image

from mmaction.registry import TRANSFORMS
from ..hardvs_paths import hardvs_frame_info

DEFAULT_DENSITY_CACHE_FILENAME = '.event_density.npy'

_warned_missing_cache = set()
_warned_stale_cache = set()
_warned_missing_sibling = set()


def event_density_from_png(img_path: str, white_threshold: int = 245) -> float:
    """Approximate event density as the fraction of non-white pixels.

    White ~= absence of an event; red/blue ~= positive/negative event.
    """
    img = np.asarray(Image.open(img_path).convert('RGB'))
    non_white = np.any(img < white_threshold, axis=-1)
    return float(non_white.mean())


def compute_sequence_densities(frame_dir: str,
                               frame_list: Sequence[str],
                               white_threshold: int = 245) -> np.ndarray:
    """Compute per-frame density in the exact order of ``frame_list``."""
    densities = np.array([
        event_density_from_png(osp.join(frame_dir, name), white_threshold)
        for name in frame_list
    ], dtype=np.float32)
    return densities


def load_or_compute_densities(
        frame_dir: str,
        frame_list: Sequence[str],
        cache_filename: str = DEFAULT_DENSITY_CACHE_FILENAME,
        white_threshold: int = 245) -> np.ndarray:
    """Load a precomputed density cache, or fall back to an in-memory
    recompute (never writes to disk from here -- see
    tools/data/hardvs/precompute_event_density.py for the offline writer, which
    avoids DataLoader-worker write races).
    """
    cache_path = osp.join(frame_dir, cache_filename)

    if osp.exists(cache_path):
        arr = np.load(cache_path)
        if arr.shape[0] == len(frame_list):
            return arr.astype(np.float32)
        if frame_dir not in _warned_stale_cache:
            _warned_stale_cache.add(frame_dir)
            print(f'[EventDensityAwareSampleFrames] Stale density cache at '
                 f'{cache_path} (len={arr.shape[0]} vs '
                 f'{len(frame_list)} frames) -- recomputing in memory. '
                 f'Re-run tools/data/hardvs/precompute_event_density.py to refresh it.')

    if frame_dir not in _warned_missing_cache:
        _warned_missing_cache.add(frame_dir)
        print(f'[EventDensityAwareSampleFrames] No density cache for '
             f'{frame_dir} -- recomputing in memory every access is slow. '
             f'Run tools/data/hardvs/precompute_event_density.py once beforehand.')

    return compute_sequence_densities(frame_dir, frame_list, white_threshold)


def _sample_from_group(indices: np.ndarray, densities: np.ndarray, k: int,
                       rng, mode: str = 'density') -> np.ndarray:
    indices = np.asarray(indices)

    if k <= 0 or len(indices) == 0:
        return np.array([], dtype=int)

    k = min(k, len(indices))

    if mode == 'density':
        local_density = densities[indices].astype(np.float64)
        p = local_density / local_density.sum() if local_density.sum() > 0 else None
    elif mode == 'uniform':
        p = None
    else:
        raise ValueError(f'Unknown mode: {mode}')

    return rng.choice(indices, size=k, replace=False, p=p)


def event_density_aware_sampling(
        densities: np.ndarray,
        num_samples: int = 8,
        ratio: Tuple[int, int] = (6, 2),
        tau: float = 0.95,
        rng=np.random,
        low_sampling_mode: str = 'uniform') -> np.ndarray:
    """Dual-interval nucleus sampling over event density (EDS, section 4.2).

    Always returns exactly ``num_samples`` sorted, unique-where-possible
    0-based frame indices -- short sequences are padded by cyclic repeat
    (mirroring SampleFrames(out_of_bound_opt='loop')), since downstream
    FormatShape requires an exact frame count.
    """
    densities = np.asarray(densities, dtype=np.float64)
    n = len(densities)

    if n == 0:
        raise ValueError('No frames available.')

    if n <= num_samples:
        selected = np.arange(n)
        if len(selected) < num_samples:
            selected = np.resize(selected, num_samples)
        return np.sort(selected)

    if densities.sum() <= 0:
        selected = rng.choice(np.arange(n), size=num_samples, replace=False)
        return np.sort(selected)

    density_prob = densities / densities.sum()

    sorted_idx = np.argsort(-density_prob)
    cum_density = np.cumsum(density_prob[sorted_idx])

    cut = np.searchsorted(cum_density, tau, side='left') + 1
    high_set = sorted_idx[:cut]
    low_set = sorted_idx[cut:]

    n_high = int(round(num_samples * ratio[0] / sum(ratio)))
    n_low = num_samples - n_high

    selected_high = _sample_from_group(high_set, densities, n_high, rng, mode='density')
    selected_low = _sample_from_group(low_set, densities, n_low, rng, mode=low_sampling_mode)

    selected = np.concatenate([selected_high, selected_low])

    if len(selected) < num_samples:
        remaining = np.setdiff1d(np.arange(n), selected, assume_unique=False)
        extra = _sample_from_group(
            remaining, densities, num_samples - len(selected), rng, mode='density')
        selected = np.concatenate([selected, extra])

    selected = np.unique(selected)

    if len(selected) < num_samples:
        remaining = np.setdiff1d(np.arange(n), selected, assume_unique=False)
        extra = rng.choice(remaining, size=num_samples - len(selected), replace=False)
        selected = np.concatenate([selected, extra])

    return np.sort(selected)


@TRANSFORMS.register_module()
class EventDensityAwareSampleFrames(BaseTransform):
    """Sample frames from an event/DVS sequence using event-density-aware
    dual-interval nucleus sampling instead of evenly-spaced segments.

    Required Keys:

        - frame_dir
        - frame_list
        - start_index

    Added Keys:

        - frame_inds
        - clip_len
        - frame_interval
        - num_clips

    Args:
        num_clips (int): Number of frames to sample. Defaults to 8.
        ratio (tuple[int, int]): High-density : low-density sampling ratio.
            Defaults to (6, 2).
        tau (float): Cumulative density threshold defining the high-density
            set. Defaults to 0.95.
        low_sampling_mode (str): 'uniform' or 'density' sampling within the
            low-density set. Defaults to 'uniform'.
        white_threshold (int): Pixel value below which a channel counts as
            "non-white" (i.e. an event) when computing density from PNGs.
            Defaults to 245.
        density_cache_filename (str): Name of the precomputed density cache
            file expected inside ``frame_dir``. Defaults to
            '.event_density.npy'.
        seed (int, optional): If set, uses a local, seeded
            ``np.random.default_rng`` (deterministic every call -- intended
            for a future deterministic-eval variant). If ``None`` (default),
            reuses the global ``np.random`` module, matching how
            ``SampleFrames`` draws randomness in this codebase so per-epoch
            variation and run-to-run reproducibility (via mmengine's
            worker seeding) behave consistently with the rest of the
            pipeline.
        clip_len (int): Only ``clip_len=1`` (2D recognizer) is implemented.
        as_single_clip (bool): The underlying frame *selection* (EDS
            itself) never changes with this flag -- it only changes how
            the ``num_clips`` selected frames are *labeled* for
            ``FormatShape`` downstream. Defaults to False: the
            ``num_clips`` frames are reported as ``num_clips`` independent
            length-1 clips (``results['num_clips']=num_clips,
            results['clip_len']=1``), which is what ``FormatShape(
            input_format='NCHW')`` expects for a 2D per-frame backbone
            (TSM). Set True for a clip-based 3D/ViT backbone (e.g.
            UniFormerV2): the same frames are instead reported as a
            *single* clip of length ``num_clips``
            (``results['num_clips']=1, results['clip_len']=num_clips``),
            which is what ``FormatShape(input_format='NCTHW')`` expects to
            produce one ``(C, T, H, W)`` tensor instead of ``T`` separate
            ``(C, H, W)`` ones.
    """

    def __init__(self,
                num_clips: int = 8,
                ratio: Tuple[int, int] = (6, 2),
                tau: float = 0.95,
                low_sampling_mode: str = 'uniform',
                white_threshold: int = 245,
                density_cache_filename: str = DEFAULT_DENSITY_CACHE_FILENAME,
                seed: Optional[int] = None,
                clip_len: int = 1,
                as_single_clip: bool = False) -> None:
        if clip_len != 1:
            raise NotImplementedError(
                'EventDensityAwareSampleFrames only supports clip_len=1 '
                '(2D recognizer) for now.')

        self.num_clips = num_clips
        self.ratio = tuple(ratio)
        self.tau = tau
        self.low_sampling_mode = low_sampling_mode
        self.white_threshold = white_threshold
        self.density_cache_filename = density_cache_filename
        self.clip_len = clip_len
        self.as_single_clip = as_single_clip
        self._rng = np.random if seed is None else np.random.default_rng(seed)

    def transform(self, results: dict) -> dict:
        frame_dir = results['frame_dir']
        frame_list = results['frame_list']
        start_index = results.get('start_index', 1)

        densities = load_or_compute_densities(
            frame_dir, frame_list, self.density_cache_filename,
            self.white_threshold)

        selected0based = event_density_aware_sampling(
            densities,
            num_samples=self.num_clips,
            ratio=self.ratio,
            tau=self.tau,
            rng=self._rng,
            low_sampling_mode=self.low_sampling_mode)

        results['frame_inds'] = (selected0based + start_index).astype(np.int32)
        results['frame_interval'] = 1
        if self.as_single_clip:
            results['clip_len'] = self.num_clips
            results['num_clips'] = 1
        else:
            results['clip_len'] = self.clip_len
            results['num_clips'] = self.num_clips
        return results

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
               f'num_clips={self.num_clips}, '
               f'ratio={self.ratio}, '
               f'tau={self.tau}, '
               f'low_sampling_mode={self.low_sampling_mode!r}, '
               f'as_single_clip={self.as_single_clip})')


@TRANSFORMS.register_module()
class PairedEventDensitySampleFrames(BaseTransform):
    """EDS sampling for an RGB pipeline, driven by the sibling DVS
    sequence's event density.

    The "non-white pixel" density heuristic used by
    ``EventDensityAwareSampleFrames`` only makes sense on event PNGs (white
    background = no event) -- applying it directly to natural RGB frames
    would be meaningless. Instead, this transform computes density from the
    paired DVS sequence (``<sequence_dir>/<sequence_name>_<dvs_suffix>``,
    resolved via the HARDVS APS/DVS sibling convention), runs the same EDS
    algorithm on that timeline, and proportionally remaps the selected
    temporal positions onto the RGB sequence's own frame count. Only the
    *timing* comes from the event modality; RGB pixels are never touched by
    the density calculation.

    Required Keys:

        - frame_dir (an HARDVS '..._<rgb_suffix>' directory)
        - total_frames (RGB total frame count)

    Added Keys:

        - frame_inds
        - clip_len
        - frame_interval
        - num_clips

    Args:
        num_clips, ratio, tau, low_sampling_mode, white_threshold,
        density_cache_filename, seed, clip_len: same as
        ``EventDensityAwareSampleFrames``.
        rgb_suffix (str): RGB sibling folder suffix. Defaults to 'aps'.
        dvs_suffix (str): event sibling folder suffix whose density drives
            sampling. Defaults to 'dvs'.
    """

    def __init__(self,
                num_clips: int = 8,
                ratio: Tuple[int, int] = (6, 2),
                tau: float = 0.95,
                low_sampling_mode: str = 'uniform',
                white_threshold: int = 245,
                density_cache_filename: str = DEFAULT_DENSITY_CACHE_FILENAME,
                rgb_suffix: str = 'aps',
                dvs_suffix: str = 'dvs',
                seed: Optional[int] = None,
                clip_len: int = 1) -> None:
        if clip_len != 1:
            raise NotImplementedError(
                'PairedEventDensitySampleFrames only supports clip_len=1 '
                '(2D recognizer) for now.')

        self.num_clips = num_clips
        self.ratio = tuple(ratio)
        self.tau = tau
        self.low_sampling_mode = low_sampling_mode
        self.white_threshold = white_threshold
        self.density_cache_filename = density_cache_filename
        self.rgb_suffix = rgb_suffix
        self.dvs_suffix = dvs_suffix
        self.clip_len = clip_len
        self._rng = np.random if seed is None else np.random.default_rng(seed)

    def _fallback_uniform(self, results: dict, rgb_total_frames: int,
                          reason: str, rgb_frame_dir: str) -> dict:
        """Sample directly on the RGB timeline (no DVS signal available)."""
        if rgb_frame_dir not in _warned_missing_sibling:
            _warned_missing_sibling.add(rgb_frame_dir)
            print(f'[{self.__class__.__name__}] {reason} for '
                 f'{rgb_frame_dir!r} -- falling back to uniform sampling '
                 f'directly on the RGB timeline for this sequence.')

        selected0based = event_density_aware_sampling(
            np.ones(rgb_total_frames, dtype=np.float32),
            num_samples=self.num_clips,
            ratio=self.ratio,
            tau=self.tau,
            rng=self._rng,
            low_sampling_mode=self.low_sampling_mode)

        start_index = results.get('start_index', 1)
        results['frame_inds'] = (selected0based + start_index).astype(np.int32)
        results['clip_len'] = self.clip_len
        results['frame_interval'] = 1
        results['num_clips'] = self.num_clips
        return results

    def transform(self, results: dict) -> dict:
        rgb_frame_dir = results['frame_dir']
        rgb_total_frames = results['total_frames']

        rgb_name = osp.basename(osp.normpath(rgb_frame_dir))
        if not rgb_name.endswith(f'_{self.rgb_suffix}'):
            raise ValueError(
                f'{self.__class__.__name__} expects an HARDVS '
                f'"..._{self.rgb_suffix}" RGB frame_dir, got '
                f'{rgb_frame_dir!r}. This transform must only be used on '
                f'the RGB training pipeline.')
        sequence_dir = osp.dirname(osp.normpath(rgb_frame_dir))

        try:
            dvs_info = hardvs_frame_info(sequence_dir, self.dvs_suffix, 'Event')
        except FileNotFoundError:
            return self._fallback_uniform(
                results, rgb_total_frames,
                'No sibling DVS sequence found', rgb_frame_dir)

        # HARDVS APS/DVS sequences almost always share the same frame count
        # (verified across the full train_label_aps.csv, 0 mismatches), so
        # DVS-selected indices normally apply directly to the RGB timeline.
        # A handful of sequences have incomplete/corrupted extractions on
        # one side only -- fall back rather than crash training on those.
        if dvs_info['total_frames'] != rgb_total_frames:
            return self._fallback_uniform(
                results, rgb_total_frames,
                f'Frame count mismatch (RGB={rgb_total_frames}, '
                f'DVS={dvs_info["total_frames"]})', rgb_frame_dir)

        densities = load_or_compute_densities(
            dvs_info['frame_dir'], dvs_info['frame_list'],
            self.density_cache_filename, self.white_threshold)

        selected0based = event_density_aware_sampling(
            densities,
            num_samples=self.num_clips,
            ratio=self.ratio,
            tau=self.tau,
            rng=self._rng,
            low_sampling_mode=self.low_sampling_mode)

        rgb_frame_inds = selected0based + dvs_info['start_index']

        results['frame_inds'] = rgb_frame_inds.astype(np.int32)
        results['clip_len'] = self.clip_len
        results['frame_interval'] = 1
        results['num_clips'] = self.num_clips
        return results

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
               f'num_clips={self.num_clips}, '
               f'ratio={self.ratio}, '
               f'tau={self.tau}, '
               f'low_sampling_mode={self.low_sampling_mode!r}, '
               f'dvs_suffix={self.dvs_suffix!r})')
