import json

import torch
from mmengine import Config
from mmengine.runner import Runner
from mmengine.runner.checkpoint import load_checkpoint
from mmseg.registry import MODELS
from mmseg.utils import register_all_modules


def main():
    register_all_modules(init_default_scope=True)
    import projects.pcb_conductor
    cfg = Config.fromfile('projects/pcb_conductor/configs/segformer_mt_vb_b.py')
    cfg.train_dataloader.num_workers = 0
    cfg.train_dataloader.persistent_workers = False

    loader = Runner.build_dataloader(cfg.train_dataloader)
    model = MODELS.build(cfg.model)
    ckpt = r'work_dirs/continual_experiment/stage2_train_domain_a/best_mmseg_mDice_iter_3000.pth'
    load_checkpoint(model, ckpt, map_location='cpu')
    model.init_old_model(ckpt)
    model.train()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    results = []
    loader_iter = iter(loader)
    for batch_idx in range(20):
        batch = next(loader_iter)
        inputs = batch['inputs']
        if isinstance(inputs, (list, tuple)):
            inputs = torch.stack(inputs, dim=0)
        data_samples = batch['data_samples']
        inputs = inputs.to(device)
        losses = model.loss(inputs, data_samples)
        is_labeled = model._get_is_labeled(data_samples, device=device).long()

        results.append({
            'batch_idx': batch_idx,
            'is_labeled': is_labeled.detach().cpu().tolist(),
            'loss_unsup': float(losses.get('loss_unsup', torch.tensor(0.0)).detach().cpu()),
            'loss_skel': float(losses.get('loss_skel', torch.tensor(0.0)).detach().cpu()),
        })

    summary = {
        'num_zero_loss_unsup': sum(1 for item in results if item['loss_unsup'] == 0.0),
        'num_zero_loss_skel': sum(1 for item in results if item['loss_skel'] == 0.0),
        'results': results,
    }
    with open('diag_unsup_multi.json', 'w', encoding='utf-8') as file:
        json.dump(summary, file, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
