"""Build the small HARDVS "KD dev" subset: 56 hand-picked classes, capped
number of recordings per class and split, labels remapped to 0..55.

The classes were selected from earlier KD runs on the full dataset to keep
the cases useful for studying distillation (see SELECTED_ACTION_IDS). Only
recordings whose APS and DVS frame folders both exist and are non-empty are
kept. At most 60 / 10 / 30 recordings per class are drawn in train / val /
test, keeping the official split of each recording.

Requires pandas. Input: the full annotation CSVs written by
make_hardvs_annotations.py.

Example:
    python tools/data/hardvs/make_hardvs_kd_dev_subset.py
"""
import argparse
import os
import os.path as osp

import pandas as pd

SPLITS = ('train', 'val', 'test')
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp')

# HARDVS action ids (action_XXX folders, i.e. original label + 1).
SELECTED_ACTION_IDS = sorted(set([
    # KD strongly improves
    27, 53, 98, 76, 13, 22, 127, 144, 135, 47,
    # KD strongly degrades
    120, 48, 65, 66, 88, 99, 241, 89, 132, 32, 140, 133,
    # hard / often missed classes
    2, 3, 14, 4, 37, 38, 45, 50, 97, 52, 117, 124,
    # short sequences
    87, 233, 267, 216, 7, 5,
    # RGB worse than Event examples
    57, 112, 115, 40, 41, 213, 123, 31,
    # RGB much better than Event examples / useful teacher cases
    138, 105, 42, 238, 55, 9, 277, 163
]))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument(
        '--ann-dir',
        default='data/hardvs/annotations',
        help='directory holding {split}_label_{aps,dvs}.csv')
    parser.add_argument(
        '--data-root',
        default='data/hardvs/rawframes',
        help='HARDVS frames, used to drop recordings without frames')
    parser.add_argument(
        '--out-dir',
        default='data/hardvs/annotations/subsets/kd_dev_56cls_seed42',
        help='output directory')
    parser.add_argument(
        '--max-per-split',
        type=int,
        nargs=3,
        default=(60, 10, 30),
        metavar=('TRAIN', 'VAL', 'TEST'),
        help='maximum number of recordings per class in each split')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def normalize_sequence_path(path: str) -> str:
    path = str(path).replace('\\', '/').strip('/')
    if path.rsplit('/', 1)[-1].lower().endswith(('_aps', '_dvs')):
        path = path.rsplit('/', 1)[0]
    return path


def read_ann(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=['path', 'label'])
    df['path'] = df['path'].astype(str).map(normalize_sequence_path)
    df['label'] = df['label'].astype(int)
    df['action_id'] = df['path'].str.extract(r'action_(\d+)')[0].astype(int)
    return df


def has_frames(data_root: str, sequence_path: str, modality: str) -> bool:
    seq_dir = osp.join(data_root, sequence_path)
    frame_dir = osp.join(seq_dir, f'{osp.basename(seq_dir)}_{modality}')
    if not osp.isdir(frame_dir):
        return False
    return any(
        name.lower().endswith(IMAGE_EXTENSIONS)
        and osp.isfile(osp.join(frame_dir, name))
        for name in os.listdir(frame_dir))


def with_modality(sequence_path: str, modality: str) -> str:
    name = sequence_path.rsplit('/', 1)[-1]
    return f'{sequence_path}/{name}_{modality}'


def main():
    args = parse_args()
    max_per_split = dict(zip(SPLITS, args.max_per_split))
    action_to_new_label = {
        action_id: new_label
        for new_label, action_id in enumerate(SELECTED_ACTION_IDS)
    }
    os.makedirs(args.out_dir, exist_ok=True)
    print(f'{len(SELECTED_ACTION_IDS)} classes -> {args.out_dir}')

    for split in SPLITS:
        aps = read_ann(osp.join(args.ann_dir, f'{split}_label_aps.csv'))
        dvs = read_ann(osp.join(args.ann_dir, f'{split}_label_dvs.csv'))
        common = aps.merge(
            dvs,
            on=['path', 'label', 'action_id'],
            how='inner',
            suffixes=('_aps', '_dvs'))
        common = common[common['action_id'].isin(SELECTED_ACTION_IDS)]

        valid = common[[
            has_frames(args.data_root, path, 'aps')
            and has_frames(args.data_root, path, 'dvs')
            for path in common['path']
        ]]

        subset = pd.concat([
            group.sample(
                n=min(len(group), max_per_split[split]),
                random_state=args.seed)
            for _, group in valid.groupby('action_id')
        ], ignore_index=True)
        subset = subset.sample(
            frac=1.0, random_state=args.seed).reset_index(drop=True)
        subset['new_label'] = subset['action_id'].map(action_to_new_label)

        for modality in ('aps', 'dvs'):
            out = pd.DataFrame({
                'path':
                subset['path'].map(lambda p: with_modality(p, modality)),
                'label': subset['new_label'],
            })
            out.to_csv(
                osp.join(args.out_dir, f'{split}_label_{modality}.csv'),
                header=False,
                index=False)
        counts = subset.groupby('action_id').size()
        print(f'  {split:5s}: {len(subset):5d} recordings | per class '
              f'min {counts.min()} / max {counts.max()} | '
              f'{len(common) - len(valid)} dropped without frames')

    pd.DataFrame([
        dict(
            new_label=new_label,
            action_id=action_id,
            action=f'action_{action_id:03d}',
            old_label=action_id - 1)
        for action_id, new_label in action_to_new_label.items()
    ]).to_csv(
        osp.join(args.out_dir, 'class_mapping.csv'), index=False)
    print('written: {split}_label_{aps,dvs}.csv, class_mapping.csv')


if __name__ == '__main__':
    main()
