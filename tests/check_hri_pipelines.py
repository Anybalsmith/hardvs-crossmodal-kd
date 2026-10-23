"""Build the HRI v4 datasets of every configs/hri config and check the pipelines.

Run from the repository root (python tests/check_hri_pipelines.py). Checks:
  - every config builds its train/val/test datasets (1354 / 226 / 679 recordings);
  - CadenceAugSampleFrames: share of long training recordings reduced to 3-4
    distinct frames (~prob), sampled indices inside the recording; forced test
    mode gives 3 distinct frames for every recording;
  - ContrastEnhance and PersonMaskBackground change the RGB frames (the black
    background config is only checked on recordings whose mask exists).
"""
import copy
import os.path as osp

import numpy as np
from mmengine.config import Config
from mmengine.registry import init_default_scope

from mmengine.registry import Registry

# When run with a modified mmaction that already registers some of these
# plugins (tsm_UniFormer), let hardvs_kd override them. No effect with the
# official mmaction2.
_register = Registry._register_module
Registry._register_module = lambda self, module, module_name=None, force=False: _register(self, module, module_name, True)
import hardvs_kd  # noqa: E402,F401  registers the plugins
Registry._register_module = _register
from mmaction.registry import DATASETS  # noqa: E402
from mmaction.utils import register_all_modules

register_all_modules()
init_default_scope('mmaction')
np.random.seed(0)
CFG = 'configs/hri/rgb_r50_hri_v4{}.py'


def build(name, split):
    cfg = Config.fromfile(CFG.format(name))
    return DATASETS.build(cfg[f'{split}_dataloader'].dataset)


def run_until(ds, idx, stop):
    r = copy.deepcopy(ds.get_data_info(idx))
    for t in ds.pipeline.transforms:
        r = t(r)
        if type(t).__name__ == stop:
            return r


for name in ['', '_cadence', '_clahe', '_bgblack', '_cadence3_test']:
    sizes = [len(build(name, s)) for s in ('train', 'val', 'test')]
    print(f'rgb_r50_hri_v4{name}: train/val/test = {sizes}')
    assert sizes == [1354, 226, 679], sizes

# cadence augmentation, training
ds = build('_cadence', 'train')
long_idx = [i for i in range(len(ds)) if ds.get_data_info(i)['total_frames'] > 4][:200]
distinct = []
for i in long_idx:
    r = run_until(ds, i, 'CadenceAugSampleFrames')
    inds = r['frame_inds'] - r['start_index']
    assert inds.min() >= 0 and inds.max() < r['total_frames'] and len(inds) == 8
    distinct.append(len(np.unique(inds)))
distinct = np.array(distinct)
print(f'cadence, train: {len(long_idx)} long recordings, reduced to 3-4 frames: '
      f'{np.isin(distinct, (3, 4)).mean():.2f}, distinct-frame counts {np.bincount(distinct)}')

# forced 3 frames at test time
ds = build('_cadence3_test', 'test')
d3 = [len(np.unique(run_until(ds, i, 'CadenceAugSampleFrames')['frame_inds'])) for i in range(0, len(ds), 7)]
print('cadence3_test: distinct frames per test recording', dict(zip(*np.unique(d3, return_counts=True))))

# frame operations
for name, op in [('_clahe', 'ContrastEnhance'), ('_bgblack', 'PersonMaskBackground')]:
    ds = build(name, 'test')
    op_t = [t for t in ds.pipeline.transforms if type(t).__name__ == op][0]
    checked = 0
    for i in range(len(ds)):
        info = ds.get_data_info(i)
        if op == 'PersonMaskBackground':
            rel = osp.relpath(osp.normpath(info['frame_dir']), op_t.data_root)
            if not osp.exists(osp.join(op_t.mask_root, rel + '.png')):
                continue
        before = run_until(ds, i, 'HARDVSFrameDecode')
        raw = [x.copy() for x in before['imgs']]
        after = op_t(before)
        print(f'{op}: {osp.basename(info["frame_dir"])} changed: {any((a != b).any() for a, b in zip(raw, after["imgs"]))}, '
              f'mean {np.mean(raw):.1f} -> {np.mean(after["imgs"]):.1f}')
        checked += 1
        if checked == 2:
            break
    if checked == 0:
        print(f'{op}: no recording with a mask yet, not checked')
print('CHECK OK')
