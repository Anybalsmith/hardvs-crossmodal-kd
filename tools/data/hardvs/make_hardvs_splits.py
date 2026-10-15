"""Generate new train / val / test splits for HARDVS (APS and DVS kept paired).

All recordings of the three original splits are pooled, then re-split class by class:

* --mode session (default): every recording session (same class, same day, same
  hour by default) goes to a single split, so val/test sessions are never seen in
  training. A class with fewer than 3 sessions cannot fill the three splits:
  with 2 sessions it gets train + test only, with 1 session its clips are split
  at random (these classes are listed in split_info.json).
* --mode random: clips are split at random within each class (like the original
  split, sessions are shared between splits).

Input: the annotation CSVs written by make_hardvs_annotations.py (or a class
subset directory).

Example:
    python tools/data/hardvs/make_hardvs_splits.py \
        --out-dir data/hardvs/annotations/splits/session_hour_seed0
    python tools/data/hardvs/make_hardvs_splits.py --session-level day --seed 1 \
        --out-dir data/hardvs/annotations/splits/session_day_seed1
"""
import argparse
import csv
import json
import os
import os.path as osp
import random
import re
from collections import Counter, defaultdict

SPLITS = ('train', 'val', 'test')
TIMESTAMP = re.compile(r'dvSave-(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2})')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument(
        '--ann-dir',
        default='data/hardvs/annotations',
        help='directory holding {train,val,test}_label_{aps,dvs}.csv '
        '(also works on a class subset directory)')
    parser.add_argument('--out-dir', required=True, help='output directory')
    parser.add_argument('--mode', choices=('session', 'random'), default='session')
    parser.add_argument('--session-level', choices=('hour', 'day'), default='hour',
                        help='what groups clips into one session')
    parser.add_argument('--ratios', type=float, nargs=3, default=(0.6, 0.1, 0.3),
                        metavar=('TRAIN', 'VAL', 'TEST'),
                        help='target share of clips per split (default: 0.6 0.1 0.3, '
                        'close to the official split)')
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def read_ann(path):
    with open(path, newline='') as f:
        return [(row[0], int(row[1])) for row in csv.reader(f)
                if len(row) >= 2 and row[1].strip().isdigit()]


def recording_id(path):
    """Strip the trailing _aps / _dvs folder: one id per recording."""
    return path.rsplit('_', 1)[0]


def session_key(path, level):
    parts = TIMESTAMP.search(path).groups()
    return '_'.join(parts[:4] if level == 'hour' else parts[:3])


def split_counts(n, ratios):
    """Integer number of items per split, summing to n."""
    counts = [int(n * r) for r in ratios]
    for i in sorted(range(3), key=lambda i: n * ratios[i] - counts[i], reverse=True):
        if sum(counts) == n:
            break
        counts[i] += 1
    return counts


def split_sessions(sessions, ratios, rng):
    """Assign whole sessions (dict key -> clip ids) to splits, close to the ratios."""
    # Big sessions first (keeps the ratios), with a random jitter so that the
    # seed changes which sessions land in val / test.
    jitter = {k: rng.uniform(0.5, 1.5) for k in sessions}
    keys = sorted(sessions, key=lambda k: len(sessions[k]) * jitter[k], reverse=True)
    total = sum(len(v) for v in sessions.values())
    target = [r * total for r in ratios]
    assigned = {s: [] for s in range(3)}
    size = [0, 0, 0]
    allowed = (0, 1, 2) if len(keys) >= 3 else (0, 2)  # 2 sessions: train + test
    for k in keys:
        s = max(allowed, key=lambda i: target[i] - size[i])
        assigned[s].append(k)
        size[s] += len(sessions[k])
    # Every allowed split gets at least one session: move the smallest session
    # of the split holding the most sessions.
    for s in allowed:
        if not assigned[s]:
            donor = max(allowed, key=lambda i: len(assigned[i]))
            k = min(assigned[donor], key=lambda k: len(sessions[k]))
            assigned[donor].remove(k)
            assigned[s].append(k)
    return {k: s for s in assigned for k in assigned[s]}


def main():
    args = parse_args()
    if abs(sum(args.ratios) - 1) > 1e-6:
        raise ValueError('--ratios must sum to 1')
    rng = random.Random(args.seed)

    # Pool all recordings; APS and DVS must describe the same recordings.
    records = {}  # recording id -> dict(label, aps, dvs)
    for split in SPLITS:
        aps = read_ann(osp.join(args.ann_dir, f'{split}_label_aps.csv'))
        dvs = read_ann(osp.join(args.ann_dir, f'{split}_label_dvs.csv'))
        assert [(recording_id(p), l) for p, l in aps] == \
            [(recording_id(p), l) for p, l in dvs], f'{split}: APS and DVS do not match'
        for (pa, label), (pd, _) in zip(aps, dvs):
            records[recording_id(pa)] = dict(label=label, aps=pa, dvs=pd)

    by_class = defaultdict(list)
    for rid in sorted(records):
        by_class[records[rid]['label']].append(rid)

    split_of = {}
    fallback = []
    sessions_per_split = Counter()
    for label in sorted(by_class):
        rids = by_class[label]
        sessions = defaultdict(list)
        for rid in rids:
            sessions[session_key(rid, args.session_level)].append(rid)
        if args.mode == 'random' or len(sessions) == 1:
            if args.mode == 'session':
                fallback.append(label)
            rids = rids[:]
            rng.shuffle(rids)
            n_train, n_val, _ = split_counts(len(rids), args.ratios)
            for i, rid in enumerate(rids):
                split_of[rid] = 0 if i < n_train else 1 if i < n_train + n_val else 2
            continue
        for key, s in split_sessions(sessions, args.ratios, rng).items():
            sessions_per_split[SPLITS[s]] += 1
            for rid in sessions[key]:
                split_of[rid] = s

    # Write the annotation files, sorted by label then recording.
    os.makedirs(args.out_dir, exist_ok=True)
    rows = {s: [] for s in SPLITS}
    for rid in sorted(records, key=lambda r: (records[r]['label'], r)):
        rows[SPLITS[split_of[rid]]].append(records[rid])
    for split in SPLITS:
        for mod in ('aps', 'dvs'):
            with open(osp.join(args.out_dir, f'{split}_label_{mod}.csv'), 'w', newline='') as f:
                csv.writer(f, lineterminator='\n').writerows(
                    (r[mod], r['label']) for r in rows[split])

    # Statistics: size, classes and share of clips whose session is in train.
    train_sessions = {(r['label'], session_key(r['dvs'], args.session_level)) for r in rows['train']}
    info = dict(mode=args.mode, session_level=args.session_level, ratios=args.ratios,
                seed=args.seed, source_ann_dir=args.ann_dir, splits={},
                single_session_classes_split_at_random=fallback)
    print(f'{len(records)} recordings, {len(by_class)} classes -> {args.out_dir}')
    for split in SPLITS:
        r = rows[split]
        seen = sum((x['label'], session_key(x['dvs'], args.session_level)) in train_sessions
                   for x in r) / max(len(r), 1)
        info['splits'][split] = dict(clips=len(r), classes=len({x['label'] for x in r}),
                                     sessions=sessions_per_split[split],
                                     share_seen_session=round(seen, 4))
        print(f'  {split:5s}: {len(r):6d} clips ({len(r) / len(records):.1%}) | '
              f'{info["splits"][split]["classes"]} classes | session seen in train: {seen:.1%}')
    if fallback:
        print(f'  {len(fallback)} classes with a single session were split at random')
    with open(osp.join(args.out_dir, 'split_info.json'), 'w') as f:
        json.dump(info, f, indent=2)


if __name__ == '__main__':
    main()
