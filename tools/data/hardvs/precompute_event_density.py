"""Precompute the per-frame event density of HARDVS DVS recordings.

The density of an event frame is its fraction of non-white pixels. It drives
the event-density aware sampling (``EventDensityAwareSampleFrames`` and
``PairedEventDensitySampleFrames``) and is cached as ``.event_density.npy``
inside each ``<seq>_dvs`` folder. Without the cache, densities are recomputed
in memory at every access, which makes training much slower.

The densities are computed offline rather than lazily by the DataLoader
workers, since concurrent workers writing the same cache would race. Each
file is written atomically, so the script can be re-run safely; valid caches
are skipped unless ``--overwrite`` is given.

Example (APS or DVS annotation files both work, only the recordings matter):
    python tools/data/hardvs/precompute_event_density.py \\
        --ann-files data/hardvs/annotations/subsets/classes_100_seed0/*_dvs.csv \\
        --data-root data/hardvs/rawframes --num-workers 8
"""
import argparse
import os
import os.path as osp
from multiprocessing import Pool

import numpy as np

from hardvs_kd.datasets.hardvs_paths import (hardvs_frame_info, read_ann_csv,
                                             sequence_dir_from_ann_path)
from hardvs_kd.datasets.transforms.event_density_sampling import (
    DEFAULT_DENSITY_CACHE_FILENAME, compute_sequence_densities)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument(
        '--ann-files',
        nargs='+',
        required=True,
        help='one or more path,label annotation CSVs')
    parser.add_argument(
        '--data-root',
        required=True,
        help='root the CSV paths are relative to (data_root in the configs)')
    parser.add_argument(
        '--white-threshold',
        type=int,
        default=245,
        help='a pixel with a channel below this value counts as an event')
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='recompute even if a valid cache already exists')
    parser.add_argument(
        '--cache-filename',
        default=DEFAULT_DENSITY_CACHE_FILENAME,
        help='file name written inside each <seq>_dvs folder')
    return parser.parse_args()


def collect_event_dirs(ann_files, data_root):
    """Return the sorted, deduplicated (frame_dir, frame_list) pairs of the
    DVS folders of every recording listed in ``ann_files``."""
    seen = {}
    missing = 0
    for ann_file in ann_files:
        for rel_path, _ in read_ann_csv(ann_file):
            sequence_dir = osp.join(data_root,
                                    sequence_dir_from_ann_path(rel_path))
            try:
                info = hardvs_frame_info(sequence_dir, 'dvs', 'Event')
            except FileNotFoundError:
                missing += 1
                continue
            seen[info['frame_dir']] = tuple(info['frame_list'])
    return sorted(seen.items()), missing


def _cache_is_valid(frame_dir, num_frames, cache_filename):
    cache_path = osp.join(frame_dir, cache_filename)
    if not osp.exists(cache_path):
        return False
    try:
        arr = np.load(cache_path)
    except Exception:  # noqa: BLE001 - unreadable cache is recomputed
        return False
    return arr.shape[0] == num_frames


def _process_one(task):
    frame_dir, frame_list, white_threshold, cache_filename, overwrite = task
    try:
        if not overwrite and _cache_is_valid(frame_dir, len(frame_list),
                                             cache_filename):
            return frame_dir, 'skipped', None

        densities = compute_sequence_densities(frame_dir, frame_list,
                                               white_threshold)
        cache_path = osp.join(frame_dir, cache_filename)
        stem = osp.splitext(cache_filename)[0]
        tmp_path = osp.join(frame_dir, f'{stem}.tmp{os.getpid()}.npy')
        np.save(tmp_path, densities)
        os.replace(tmp_path, cache_path)
        return frame_dir, 'processed', None
    except Exception as exc:  # noqa: BLE001 - report, don't crash the pool
        return frame_dir, 'failed', str(exc)


def main():
    args = parse_args()

    event_dirs, missing = collect_event_dirs(args.ann_files, args.data_root)
    print(f'Found {len(event_dirs)} unique DVS recordings in '
          f'{len(args.ann_files)} annotation file(s) '
          f'({missing} annotation rows without a DVS folder).')

    tasks = [(frame_dir, frame_list, args.white_threshold,
              args.cache_filename, args.overwrite)
             for frame_dir, frame_list in event_dirs]

    counts = dict(processed=0, skipped=0, failed=0)
    failures = []
    with Pool(args.num_workers) as pool:
        for i, (frame_dir, status, err) in enumerate(
                pool.imap_unordered(_process_one, tasks), start=1):
            counts[status] += 1
            if status == 'failed':
                failures.append((frame_dir, err))
            if i % 500 == 0 or i == len(tasks):
                print(f'[{i}/{len(tasks)}] ' +
                      ' '.join(f'{k}={v}' for k, v in counts.items()))

    print('Done. ' + ' '.join(f'{k}={v}' for k, v in counts.items()))
    if failures:
        print('Failures:')
        for frame_dir, err in failures[:20]:
            print(f'  {frame_dir}: {err}')
        if len(failures) > 20:
            print(f'  ... and {len(failures) - 20} more')


if __name__ == '__main__':
    main()
