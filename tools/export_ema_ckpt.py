import argparse
import torch

def pick_ema_state_dict(ckpt: dict):
    """Try best to find EMA/teacher weights in checkpoint."""
    # 1) common direct fields
    for k in ['ema_state_dict', 'state_dict_ema', 'teacher_state_dict']:
        if k in ckpt and isinstance(ckpt[k], dict):
            return ckpt[k]

    sd = ckpt.get('state_dict', ckpt)
    if not isinstance(sd, dict):
        raise ValueError("No state_dict found in checkpoint.")

    # 2) prefixed keys inside state_dict
    prefixes = ['ema_model.', 'ema.', 'teacher.', 'model_ema.', 'ema_teacher.']
    for pref in prefixes:
        if any(key.startswith(pref) for key in sd.keys()):
            ema_sd = {key[len(pref):]: val for key, val in sd.items() if key.startswith(pref)}
            if len(ema_sd) > 0:
                return ema_sd

    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', required=True)
    ap.add_argument('--out', dest='out', required=True)
    args = ap.parse_args()

    ckpt = torch.load(args.inp, map_location='cpu')
    ema_sd = pick_ema_state_dict(ckpt)

    if ema_sd is None:
        raise RuntimeError(
            "EMA/teacher weights not found in this checkpoint. "
            "You may not be saving EMA, or key names differ."
        )

    # write a normal mmengine-style ckpt with EMA as state_dict
    new_ckpt = dict(
        meta=ckpt.get('meta', {}),
        state_dict=ema_sd,
        message='Exported EMA/teacher weights as inference state_dict'
    )
    torch.save(new_ckpt, args.out)
    print(f"[OK] saved EMA inference ckpt: {args.out}")
    print(f"[OK] num params: {len(ema_sd)}")

if __name__ == '__main__':
    main()
