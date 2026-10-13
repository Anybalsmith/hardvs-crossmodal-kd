import torch
from mmengine.dataset import pseudo_collate
from mmengine.registry import FUNCTIONS


@FUNCTIONS.register_module()
def crossmodal_collate(data_batch):
    """``pseudo_collate`` plus stacking of the teacher inputs.

    ``HARDVSPairedDataset`` adds an ``inputs_teacher`` tensor next to the
    student ``inputs``; it is stacked into a batch tensor here.
    """
    if not data_batch:
        return None

    collated = pseudo_collate(data_batch)
    if 'inputs_teacher' in data_batch[0]:
        collated['inputs_teacher'] = torch.stack(
            [item['inputs_teacher'] for item in data_batch], dim=0)
    return collated
