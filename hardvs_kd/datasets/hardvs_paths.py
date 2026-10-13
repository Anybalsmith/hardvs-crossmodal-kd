"""HARDVS path helpers shared by the datasets and the density tools.

A HARDVS recording is stored as::

    <data_root>/action_XXX/<seq>/<seq>_aps/*.png   # RGB (APS) frames
    <data_root>/action_XXX/<seq>/<seq>_dvs/*.png   # event (DVS) frames

Annotation CSVs list ``action_XXX/<seq>/<seq>_{aps,dvs},<label>`` rows.
"""
import os
import os.path as osp
import re
from typing import Sequence

import numpy as np


def sequence_dir_from_ann_path(rel_path: str) -> str:
    """Strip the trailing ``<seq>_aps`` / ``<seq>_dvs`` folder of an
    annotation path, so that both modalities map to the same recording."""
    rel_path = rel_path.replace('\\', '/').strip('/')
    if rel_path.rsplit('/', 1)[-1].lower().endswith(('_aps', '_dvs')):
        rel_path = rel_path.rsplit('/', 1)[0]
    return rel_path


def read_ann_csv(ann_file: str) -> list:
    """Return the ``(rel_path, label)`` rows of a ``path,label`` CSV."""
    samples = []
    with open(ann_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) != 2:
                raise ValueError(
                    f'Invalid annotation line in {ann_file}: {line}')
            samples.append((parts[0].strip(), int(parts[1].strip())))
    return samples


def hardvs_frame_info(sequence_dir: str, suffix: str, modality: str) -> dict:
    """Resolve ``<sequence_dir>/<sequence_name>_<suffix>`` and describe its
    numerically-sorted frames.
    """
    sequence_name = osp.basename(osp.normpath(sequence_dir))
    frame_dir = osp.join(sequence_dir, f'{sequence_name}_{suffix}')
    if not osp.isdir(frame_dir):
        raise FileNotFoundError(frame_dir)

    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp')
    files = [
        name for name in os.listdir(frame_dir)
        if name.lower().endswith(valid_exts)
    ]

    def key(name):
        numbers = re.findall(r'\d+', osp.splitext(name)[0])
        return int(numbers[-1]) if numbers else 0

    files.sort(key=key)
    if not files:
        raise FileNotFoundError(f'No image frames found in {frame_dir}')

    return dict(
        frame_dir=frame_dir,
        total_frames=len(files),
        frame_list=files,
        filename_tmpl='{:08d}.png',
        start_index=1,
        modality=modality,
    )


def remap_frame_inds(source_inds: Sequence[int], source_total_frames: int,
                     target_total_frames: int):
    """Proportionally rescale 1-based frame indices from one modality's
    timeline onto another's.
    """
    if target_total_frames <= 0:
        raise ValueError('target_total_frames must be > 0')

    frame_inds = np.asarray(source_inds)

    if source_total_frames <= 1 or target_total_frames == 1:
        return frame_inds * 0 + 1

    remapped = np.round(
        (frame_inds - 1) * (target_total_frames - 1) /
        (source_total_frames - 1)).astype(frame_inds.dtype) + 1
    return np.clip(remapped, 1, target_total_frames)
