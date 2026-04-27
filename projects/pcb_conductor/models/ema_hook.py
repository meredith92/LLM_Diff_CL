from mmengine.hooks import Hook
from mmseg.registry import HOOKS

@HOOKS.register_module()
class EMAUpdateHook(Hook):
    def after_train_iter(self, runner, batch_idx: int, data_batch=None, outputs=None):
        model = runner.model
        if hasattr(model, 'ema_update'):
            model.ema_update()
