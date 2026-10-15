"""Convert the HARDVS split lists into the per-modality annotation CSVs used
by the configs.

Input, one line per recording (official HARDVS ``{split}_label.txt``)::

    action_001/dvSave-2021_08_20_16_15_22 4 0      # path num_frames label

Output, ``{split}_label_aps.csv`` (RGB) and ``{split}_label_dvs.csv``
(event)::

    action_001/dvSave-2021_08_20_16_15_22/dvSave-2021_08_20_16_15_22_aps,0

Example:
    python tools/data/hardvs/make_hardvs_annotations.py \\
        --label-dir data/hardvs --out-dir data/hardvs/annotations
"""
import argparse
import csv
import os
import os.path as osp

SPLITS = ('train', 'val', 'test')
MODALITIES = ('aps', 'dvs')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument(
        '--label-dir',
        default='data/hardvs',
        help='directory holding {train,val,test}_label.txt')
    parser.add_argument(
        '--out-dir',
        default='data/hardvs/annotations',
        help='output directory of {split}_label_{aps,dvs}.csv')
    return parser.parse_args()


def read_label_txt(path):
    """Return the (recording path, label) rows of a HARDVS split list."""
    rows = []
    with open(path) as f:
        for line_no, line in enumerate(f, start=1):
            parts = line.split()
            if not parts:
                continue
            if len(parts) != 3:
                raise ValueError(f'{path}:{line_no}: expected '
                                 f'"path num_frames label", got {line!r}')
            rows.append((parts[0].strip('/'), int(parts[2])))
    return rows


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    for split in SPLITS:
        rows = read_label_txt(osp.join(args.label_dir, f'{split}_label.txt'))
        for modality in MODALITIES:
            out_path = osp.join(args.out_dir, f'{split}_label_{modality}.csv')
            with open(out_path, 'w', newline='') as f:
                writer = csv.writer(f, lineterminator='\n')
                for path, label in rows:
                    name = path.rsplit('/', 1)[-1]
                    writer.writerow([f'{path}/{name}_{modality}', label])
        num_classes = len({label for _, label in rows})
        print(f'{split:5s}: {len(rows):6d} recordings, {num_classes} classes')
    print(f'written: {args.out_dir}/{{split}}_label_{{aps,dvs}}.csv')


if __name__ == '__main__':
    main()
