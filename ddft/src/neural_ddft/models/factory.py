import torch
import torch.nn as nn

from neural_ddft.models.model import SuperadiabaticNet, OPTIONAL_CHANNELS
from neural_ddft.models.zimmermann_net import ZimmermannNet
from neural_ddft.models.neural_free_energy import NeuralFreeEnergy, NeuralFreeEnergyFine

BASE_CHECKPOINT_PATH = "models/c2_dz32.ckpt"
MODEL_KEY_BY_MODE = {"equilibrium": "base", "hybrid": "sup", "zimmermann": "zimmermann"}


def _load_base_model(base_model_path, dx, device):
    """
    Loads the equilibrium free-energy model architecture matching the grid
    resolution (fine dz=1/100 vs coarse dz=1/32), then overlays
    `base_model_path` weights on top if they differ from the architecture checkpoint.
    """
    model_cls = NeuralFreeEnergyFine if int(1 / dx) == 100 else NeuralFreeEnergy
    base_model = model_cls.load_from_checkpoint(checkpoint_path=BASE_CHECKPOINT_PATH, map_location=device)

    try:
        ckpt_data = torch.load(base_model_path, map_location=device)
        if isinstance(ckpt_data, dict) and "state_dict" in ckpt_data:
            base_model.load_state_dict(ckpt_data["state_dict"])
        elif isinstance(ckpt_data, dict):
            base_model.load_state_dict(ckpt_data)
        print(f"Successfully loaded base model weights from {base_model_path}")
    except Exception as e:
        if base_model_path != BASE_CHECKPOINT_PATH:
            print(f"Warning: Could not load state dict from {base_model_path} ({e}), falling back to direct load_from_checkpoint.")
            base_model = model_cls.load_from_checkpoint(checkpoint_path=base_model_path, map_location=device)

    return base_model.to(device)


def _load_correction_checkpoint(model, ckpt_path, device):
    if not ckpt_path:
        return
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        # Checkpoints saved since the optional-input-channel feature carry their
        # use_* flags alongside the weights - verify they match this model
        # instance before loading, so a channel-count mismatch is reported with
        # a clear reason instead of either a cryptic shape error or (worse) a
        # silent misload if the channel counts happen to coincide.
        for flag in OPTIONAL_CHANNELS:
            if flag in ckpt and getattr(model, flag, False) != ckpt[flag]:
                raise ValueError(
                    f"Checkpoint '{ckpt_path}' was trained with {flag}={ckpt[flag]} "
                    f"but the model was instantiated with {flag}={getattr(model, flag, False)}."
                )
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict)


def build_models(mode, device, dx, base_model_path=BASE_CHECKPOINT_PATH,
                  sup_model_path=None, zimmermann_model_path=None, trainable=True,
                  sup_model_kwargs=None):
    """
    Builds the {'base', 'sup'|'zimmermann'} model dict and the list of
    trainable parameters for a given force mode ("equilibrium"/"hybrid"/"zimmermann").

    trainable=True  (training scripts): equilibrium mode fine-tunes only the
        base model's final layer; hybrid/zimmermann train their correction net.
    trainable=False (evaluation scripts): every model is frozen and put in eval().
    sup_model_kwargs: optional dict of SuperadiabaticNet constructor kwargs
        (e.g. {"use_vext": True}) - only used when mode == "hybrid".
    """
    if mode not in MODEL_KEY_BY_MODE:
        raise ValueError(f"Unknown model_mode '{mode}'")

    models = {}
    trainable_params = []

    base_model = _load_base_model(base_model_path, dx, device)
    if trainable and mode == "equilibrium":
        base_model.train()
        for p in base_model.parameters():
            p.requires_grad = False
        final_layer = list(base_model.children())[-1]
        if isinstance(final_layer, nn.Sequential):
            final_layer = final_layer[-1]
        for p in final_layer.parameters():
            p.requires_grad = True
            trainable_params.append(p)
        print("Equilibrium mode: Froze feature extractors. Fine-tuning ONLY the final layer.")
    else:
        base_model.eval()
        for p in base_model.parameters():
            p.requires_grad = False
    models["base"] = base_model

    if mode == "hybrid":
        sup_model = SuperadiabaticNet(channels=16, **(sup_model_kwargs or {})).to(device)
        _load_correction_checkpoint(sup_model, sup_model_path, device)
        if trainable:
            sup_model.train()
            trainable_params.extend(sup_model.parameters())
        else:
            sup_model.eval()
        models["sup"] = sup_model
    elif mode == "zimmermann":
        z_model = ZimmermannNet().to(device)
        _load_correction_checkpoint(z_model, zimmermann_model_path, device)
        if trainable:
            z_model.train()
            trainable_params.extend(z_model.parameters())
        else:
            z_model.eval()
        models["zimmermann"] = z_model

    return models, trainable_params
