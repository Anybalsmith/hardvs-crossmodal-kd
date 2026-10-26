"""Per-frame person masks (YOLO11x-seg + SAM 2.1-L) of the TRAINING clips, for PersonBackgroundMix.

A clip is recomposed by the augmentation only if the person is found on every PNG:
  1. YOLO11x-seg on every PNG of the clip (person boxes, confidence >= 0.25);
  2. if some PNG has no detection, the clip is NOT eligible and gets no mask (always kept with its own
     scene); no clip-box fallback, which made SAM 2 segment lamps and shadows at night;
  3. otherwise SAM 2.1-L turns the YOLO boxes of each PNG into its mask (union of the persons), in half
     precision by default (2.5x faster; on a 48-frame sample the masks were identical to float32,
     IoU 1.000; --float32 to disable).
Only the training split is processed: validation and test are never recomposed.

Output (--out): <action>/<seq>/<seq>_aps/<png> masks (uint8 0/255) for eligible clips, and clips.csv
(folder, label, frames, frames_detected, eligible). Restartable: clips already listed are skipped.

Parallel run on one GPU: K processes with --shard 0..K-1 --num-shards K each take every K-th clip of
the training list and write clips_shard<i>.csv; then --merge appends them to clips.csv. On an A6000
the GPU is already saturated by one float32 process; in half precision 2 processes gave ~110 clips / 5 min.

Needs ultralytics >= 8.3 (SAM 2) and the weights yolo11x-seg.pt and sam2.1_l.pt (downloaded by
ultralytics on first use if the machine has internet, otherwise pass their paths). Run from the
repository root, e.g.
    python tools/data/hardvs/make_mix_masks.py --ann-dir data/hardvs/annotations/splits/hri_v1_official
"""
import argparse
import csv
import os
import os.path as osp

import numpy as np
import pandas as pd
from PIL import Image


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--ann-dir', required=True, help='folder with train_label_aps.csv')
    ap.add_argument('--data-root', default='data/hardvs/rawframes')
    ap.add_argument('--out', default=None, help='default: data/hardvs/person_masks/mix_sam2_<subset>')
    ap.add_argument('--yolo', default='yolo11x-seg.pt')
    ap.add_argument('--sam', default='sam2.1_l.pt')
    ap.add_argument('--batch', type=int, default=16, help='YOLO batch size')
    ap.add_argument('--device', default='0')
    ap.add_argument('--float32', action='store_true', help='run SAM 2 in float32 instead of half precision')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--num-shards', type=int, default=1)
    ap.add_argument('--merge', action='store_true', help='merge the shard files into clips.csv and stop')
    args = ap.parse_args()
    if args.out is None:
        args.out = 'data/hardvs/person_masks/mix_sam2_' + osp.basename(osp.normpath(args.ann_dir))
    return args


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    stats = osp.join(args.out, 'clips.csv')

    def shard_files():
        return sorted(f for f in os.listdir(args.out)
                      if f.startswith('clips_shard') and os.path.getsize(osp.join(args.out, f)) > 0)

    if args.merge:
        parts = ([pd.read_csv(stats)] if osp.exists(stats) else []) + \
            [pd.read_csv(osp.join(args.out, f)) for f in shard_files()]
        merged = pd.concat(parts, ignore_index=True).drop_duplicates('folder')
        merged.to_csv(stats, index=False)
        for f in shard_files():
            os.remove(osp.join(args.out, f))
        print(f'{len(merged)} clips in {stats}, eligible {merged.eligible.mean():.3f}')
        print('done')
        return

    from ultralytics import SAM, YOLO
    yolo = YOLO(args.yolo)
    sam = SAM(args.sam)
    done = set(pd.read_csv(stats).folder) if osp.exists(stats) else set()
    for f in shard_files():
        # another shard may be writing its file: read the complete lines only
        done |= {line.split(',')[0] for line in open(osp.join(args.out, f)).read().splitlines()[1:]}
    rows = list(csv.reader(open(osp.join(args.ann_dir, 'train_label_aps.csv'))))[args.shard::args.num_shards]
    rows = [r for r in rows if r[0] not in done]
    out_csv = stats if args.num_shards == 1 else osp.join(args.out, f'clips_shard{args.shard}.csv')
    new_file = not osp.exists(out_csv) or os.path.getsize(out_csv) == 0
    with open(out_csv, 'a', newline='') as fs:
        w = csv.writer(fs)
        if new_file:
            w.writerow(['folder', 'label', 'frames', 'frames_detected', 'eligible'])
        for n, (folder, label) in enumerate(rows):
            pngs = sorted(os.listdir(osp.join(args.data_root, folder)))
            imgs = [np.asarray(Image.open(osp.join(args.data_root, folder, p)).convert('RGB')) for p in pngs]
            boxes = []
            for i in range(0, len(imgs), args.batch):
                det = yolo.predict([im[..., ::-1] for im in imgs[i:i + args.batch]], imgsz=640, conf=0.25,
                                   classes=[0], device=args.device, verbose=False)
                boxes += [r.boxes.xyxy.cpu().numpy() for r in det]
            detected = sum(len(b) > 0 for b in boxes)
            eligible = detected == len(pngs)
            if eligible:
                d = osp.join(args.out, folder)
                os.makedirs(d, exist_ok=True)
                for p, im, b in zip(pngs, imgs, boxes):
                    r = sam(im[..., ::-1], bboxes=b.tolist(), device=args.device, half=not args.float32,
                            verbose=False)[0]
                    m = r.masks.data.cpu().numpy().astype(bool).any(0) if r.masks is not None \
                        else np.zeros(im.shape[:2], bool)
                    Image.fromarray(m.astype(np.uint8) * 255).save(osp.join(d, p))
            w.writerow([folder, label, len(pngs), detected, int(eligible)])
            fs.flush()
            if n % 50 == 0:
                print(f'shard {args.shard}: {n} / {len(rows)}', flush=True)
    print(f'shard {args.shard} finished')


if __name__ == '__main__':
    main()
