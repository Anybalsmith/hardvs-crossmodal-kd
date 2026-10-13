import json
import os.path as osp

from mmengine.hooks import Hook
from mmaction.registry import HOOKS


@HOOKS.register_module()
class TestDuringTrainHook(Hook):
    """Run the test loop after validation every `interval` epochs.

    Test metrics are logged and appended to ``<log_dir>/test_metrics.jsonl``.
    """
    priority = 'LOW'  # after LoggerHook, before CheckpointHook

    def __init__(self, interval=5, test_last=True):
        self.interval = interval
        self.test_last = test_last

    def after_val_epoch(self, runner, metrics=None):
        epoch = runner.epoch  # number of finished epochs
        is_last = epoch == runner.max_epochs
        if epoch % self.interval != 0 and not (self.test_last and is_last):
            return
        test_metrics = runner.test_loop.run()
        runner.visualizer.add_scalars(
            {f'test/{k}': v for k, v in test_metrics.items()}, step=epoch)
        with open(osp.join(runner.log_dir, 'test_metrics.jsonl'), 'a') as f:
            f.write(json.dumps({'epoch': epoch, **test_metrics}) + '\n')
        runner.logger.info(f'[Test @ epoch {epoch}] {test_metrics}')
