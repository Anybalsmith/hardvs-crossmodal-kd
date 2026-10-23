"""Build the HRI (human-robot / drone command) class subset of HARDVS.

The classes come from an explicit, hand-made list (``--classes``, a CSV with
one row per kept HARDVS class: ``action`` = HARDVS action number 1..300,
``hri_command`` and ``sources``). The splits are the pipeline's
``{split}_label_{aps,dvs}.csv``; the clips are not resampled, only filtered,
and the labels are remapped to 0..K-1 in the order of the list.

Output, same layout as ``subsets/classes_50_seed0``:
    {split}_label_{aps,dvs}.csv   path,label (no header)
    classes.csv                   new_label,orig_label,action_folder,hardvs_name,hri_command
    subset_info.json              list, counts per split and per class

Run from the repository root, after make_hardvs_annotations.py (or with
--ann-dir pointing to another folder of {split}_label_{aps,dvs}.csv). Class
names are read from --class-names (HARDVS_300_class.txt of the HARDVS
release) when the file exists, otherwise left empty. Example:
    python tools/data/hardvs/make_hri_subset.py \
        --classes tools/data/hardvs/hri_classes/hri_classes_v4.csv \
        --out-dir data/hardvs/annotations/splits/hri_v4_official
"""
import argparse
import csv
import json
import os
import os.path as osp
from collections import Counter

ANN_DIR = 'data/hardvs/annotations'
CLASS_NAMES = 'data/hardvs/HARDVS_300_class.txt'
SPLITS = ('train', 'val', 'test')
MODALITIES = ('aps', 'dvs')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--classes', required=True, help='CSV of the kept classes')
    parser.add_argument('--ann-dir', default=ANN_DIR)
    parser.add_argument('--class-names', default=CLASS_NAMES)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def read_ann(path):
    with open(path, newline='') as f:
        return [(row[0], int(row[1])) for row in csv.reader(f) if len(row) >= 2]


def hardvs_names(path):
    names = {}
    if not osp.exists(path):
        return names
    for line in open(path):
        line = line.strip()
        if line.lower().startswith('action'):
            num, _, rest = line.partition(' ')
            names[int(num[6:9])] = rest.strip(' -')
    return names


def main():
    args = parse_args()
    names = hardvs_names(args.class_names)
    with open(args.classes, newline='') as f:
        kept = list(csv.DictReader(f))
    actions = [int(r['action']) for r in kept]
    assert len(set(actions)) == len(actions), 'class listed twice'
    # HARDVS label = action number - 1
    new_label = {a - 1: i for i, a in enumerate(actions)}

    anns = {(s, m): read_ann(osp.join(args.ann_dir, f'{s}_label_{m}.csv'))
            for s in SPLITS for m in MODALITIES}
    for s in SPLITS:
        aps = [(p.rsplit('_', 1)[0], l) for p, l in anns[(s, 'aps')]]
        dvs = [(p.rsplit('_', 1)[0], l) for p, l in anns[(s, 'dvs')]]
        assert aps == dvs, f'{s}: APS and DVS annotations differ'

    counts = {}
    for (s, m), rows in anns.items():
        rows = [(p, new_label[l]) for p, l in rows if l in new_label]
        if m == 'dvs':
            counts[s] = Counter(l for _, l in rows)
        if not args.dry_run:
            os.makedirs(args.out_dir, exist_ok=True)
            with open(osp.join(args.out_dir, f'{s}_label_{m}.csv'), 'w', newline='') as f:
                csv.writer(f, lineterminator='\n').writerows(rows)

    print(f'{len(actions)} classes -> {args.out_dir}')
    print(f'{"new":>3} {"A":>4} {"train":>5} {"val":>4} {"test":>4}  hardvs name -> HRI command')
    for i, r in enumerate(kept):
        a = int(r['action'])
        print(f'{i:>3} {a:>4} {counts["train"][i]:>5} {counts["val"][i]:>4} {counts["test"][i]:>4}  '
              f'{names.get(a, "")[:32]:32s} -> {r["hri_command"]}')
    print('total', {s: sum(c.values()) for s, c in counts.items()})

    if args.dry_run:
        return
    with open(osp.join(args.out_dir, 'classes.csv'), 'w', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['new_label', 'orig_label', 'action_folder', 'hardvs_name', 'hri_command'])
        for i, r in enumerate(kept):
            a = int(r['action'])
            w.writerow([i, a - 1, f'action_{a:03d}', names.get(a, ''), r['hri_command']])
    info = dict(name=osp.basename(args.out_dir.rstrip('/')), num_classes=len(actions),
                source_ann_dir=args.ann_dir, class_list=osp.abspath(args.classes),
                orig_labels=[a - 1 for a in actions],
                classes=[dict(new_label=i, action=int(r['action']), hardvs_name=names.get(int(r['action']), ''),
                              hri_command=r['hri_command'], sources=r.get('sources', ''))
                         for i, r in enumerate(kept)],
                num_samples={s: sum(c.values()) for s, c in counts.items()},
                per_class={s: [counts[s][i] for i in range(len(actions))] for s in SPLITS})
    with open(osp.join(args.out_dir, 'subset_info.json'), 'w') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
