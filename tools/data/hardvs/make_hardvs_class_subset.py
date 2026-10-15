"""Build a HARDVS class subset: draw N classes at random and write the matching
annotation files (APS and DVS, train/val/test) with labels remapped to 0..N-1.

The classes are the first N of a seeded random permutation of all classes, so
with the same seed the 50-class subset is included in the 100-class subset.

Input: the full annotation CSVs written by make_hardvs_annotations.py.

Example:
    python tools/data/hardvs/make_hardvs_class_subset.py --num-classes 50 --seed 0
    python tools/data/hardvs/make_hardvs_class_subset.py --num-classes 100 --seed 0
"""
import argparse
import csv
import json
import os
import os.path as osp
from collections import Counter

import numpy as np

SPLITS = ('train', 'val', 'test')
MODALITIES = ('aps', 'dvs')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--num-classes', type=int, required=True,
                        help='number of classes to draw')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed of the class draw')
    parser.add_argument(
        '--ann-dir',
        default='data/hardvs/annotations',
        help='directory holding {split}_label_{aps,dvs}.csv')
    parser.add_argument(
        '--out-dir', default=None,
        help='output directory (default: <ann-dir>/subsets/'
        'classes_<N>_seed<S>)')
    parser.add_argument('--dry-run', action='store_true',
                        help='print the statistics without writing files')
    return parser.parse_args()


def read_ann(path):
    """Return the (path, label) rows of a HARDVS annotation csv."""
    with open(path, newline='') as f:
        return [(row[0], int(row[1])) for row in csv.reader(f)
                if len(row) >= 2 and row[1].strip().isdigit()]


def recording_id(path):
    """Strip the trailing _aps / _dvs folder to compare both modalities."""
    return path.rsplit('_', 1)[0]


def main():
    args = parse_args()
    anns = {(s, m): read_ann(osp.join(args.ann_dir, f'{s}_label_{m}.csv'))
            for s in SPLITS for m in MODALITIES}

    # APS and DVS files must list the same recordings with the same labels.
    for s in SPLITS:
        aps, dvs = anns[(s, 'aps')], anns[(s, 'dvs')]
        assert [(recording_id(p), l) for p, l in aps] == \
            [(recording_id(p), l) for p, l in dvs], \
            f'{s}: APS and DVS annotations do not match'

    all_classes = sorted({l for rows in anns.values() for _, l in rows})
    if not 0 < args.num_classes <= len(all_classes):
        raise ValueError(f'--num-classes must be in [1, {len(all_classes)}]')

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(all_classes)
    chosen = sorted(int(c) for c in order[:args.num_classes])
    new_label = {orig: new for new, orig in enumerate(chosen)}

    out_dir = args.out_dir or osp.join(
        args.ann_dir, 'subsets',
        f'classes_{args.num_classes}_seed{args.seed}')
    print(f'{args.num_classes} classes drawn with seed {args.seed} '
          f'out of {len(all_classes)}')
    print(f'output: {out_dir}')

    summary = {}
    for (s, m), rows in anns.items():
        kept = [(p, new_label[l]) for p, l in rows if l in new_label]
        counts = Counter(l for _, l in kept)
        summary[f'{s}_{m}'] = len(kept)
        if m == 'aps':
            print(f'  {s:5s}: {len(kept):6d} samples | per class '
                  f'min {min(counts.values())} / max {max(counts.values())}'
                  f' | classes present {len(counts)}')
        if not args.dry_run:
            os.makedirs(out_dir, exist_ok=True)
            with open(osp.join(out_dir, f'{s}_label_{m}.csv'), 'w',
                      newline='') as f:
                csv.writer(f, lineterminator='\n').writerows(kept)

    if args.dry_run:
        return
    # Mapping between the new labels and the original HARDVS classes.
    with open(osp.join(out_dir, 'classes.csv'), 'w', newline='') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(['new_label', 'orig_label', 'action_folder'])
        for orig in chosen:
            writer.writerow([new_label[orig], orig, f'action_{orig + 1:03d}'])
    with open(osp.join(out_dir, 'subset_info.json'), 'w') as f:
        json.dump(dict(num_classes=args.num_classes, seed=args.seed,
                       source_ann_dir=args.ann_dir, orig_labels=chosen,
                       num_samples=summary), f, indent=2)
    print('written: {split}_label_{aps,dvs}.csv, classes.csv, '
          'subset_info.json')


if __name__ == '__main__':
    main()
