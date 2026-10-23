# HRI v4 (the 5 classes worst recognised by the v1 baselines: A072, A006,
# A073, A129, A106), official split: RGB TSM-R50 on the raw APS frames.
# Reference for the *_cadence, *_clahe and *_bgblack variants (same settings,
# same seed, only one transform differs).
_base_ = ['./_base_hri_rgb_r50.py']
