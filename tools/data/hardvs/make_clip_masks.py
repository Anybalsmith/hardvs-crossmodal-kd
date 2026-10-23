"""Clip-level person masks of the APS frames, for the PersonMaskBackground transform.

For each recording of an annotation folder: YOLO11x-seg person mask of every
APS PNG (union of the 'person' instances, confidence >= 0.25), union over the
recording, dilated by --dilate pixels. One uint8 PNG (0/255, frame size) per
recording, at <out>/<action_xxx>/<seq>/<seq>_aps.png. A recording where no
frame has a person gets an all-zero mask; the transform then leaves its frames
unchanged. Also writes <out>/clip_stats.csv. Already computed masks are
skipped, so the script can be stopped and restarted.

Needs ultralytics (pip install ultralytics) and the YOLO11x-seg weights
(https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x-seg.pt).
Run from the repository root:
    python tools/data/hardvs/make_clip_masks.py \
        --ann-dir data/hardvs/annotations/splits/hri_v4_official \
        --weights /path/to/yolo11x-seg.pt
"""
import argparse
import csv
import os
import os.path as osp

import cv2
import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--ann-dir', required=True,
                        help='folder of {train,val,test}_label_aps.csv')
    parser.add_argument('--data-root', default='data/hardvs/rawframes')
    parser.add_argument('--out', default='data/hardvs/person_masks/clip_masks_yolo_d15')
    parser.add_argument('--weights', required=True)
    parser.add_argument('--dilate', type=int, default=15)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--device', default='cuda')
    return parser.parse_args()


def main():
    args = parse_args()
    from ultralytics import YOLO
    model = YOLO(args.weights)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * args.dilate + 1, 2 * args.dilate + 1))

    def person_masks(imgs):
        # ultralytics expects BGR arrays; retina_masks gives masks at frame size
        res = model.predict([im[..., ::-1] for im in imgs], imgsz=640, conf=0.25, classes=[0],
                            retina_masks=True, device=args.device, verbose=False)
        return [np.zeros(im.shape[:2], bool) if r.masks is None
                else r.masks.data.cpu().numpy().astype(bool).any(0)
                for im, r in zip(imgs, res)]

    stats_path = osp.join(args.out, 'clip_stats.csv')
    os.makedirs(args.out, exist_ok=True)
    new_file = not osp.exists(stats_path)
    with open(stats_path, 'a', newline='') as fstats:
        writer = csv.writer(fstats)
        if new_file:
            writer.writerow(['split', 'folder', 'label', 'frames', 'frames_person', 'area'])
        for split in ('train', 'val', 'test'):
            ann = osp.join(args.ann_dir, f'{split}_label_aps.csv')
            rows = [r for r in csv.reader(open(ann)) if len(r) >= 2]
            for n, row in enumerate(rows):
                folder, label = row[0], row[-1]
                out = osp.join(args.out, folder + '.png')
                if osp.exists(out):
                    continue
                d = osp.join(args.data_root, folder)
                pngs = sorted(os.listdir(d))
                union, found = None, 0
                for i in range(0, len(pngs), args.batch):
                    imgs = [np.asarray(Image.open(osp.join(d, p)).convert('RGB'))
                            for p in pngs[i:i + args.batch]]
                    for m in person_masks(imgs):
                        found += int(m.any())
                        union = m if union is None else (union | m)
                mask = cv2.dilate(union.astype(np.uint8), kernel) > 0
                os.makedirs(osp.dirname(out), exist_ok=True)
                Image.fromarray(mask.astype(np.uint8) * 255).save(out)
                writer.writerow([split, folder, label, len(pngs), found, f'{mask.mean():.5f}'])
                fstats.flush()
                if n % 200 == 0:
                    print(split, n, '/', len(rows), flush=True)
    print('done')


if __name__ == '__main__':
    main()
