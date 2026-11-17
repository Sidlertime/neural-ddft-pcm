import torch
import torch.nn as nn

# Optional input channels, in the fixed order they're stacked when enabled.
# Each flag name matches a SuperadiabaticNet constructor arg and a forward()
# keyword arg of the same name (minus "use_"). "use_grad" is the one
# exception to the "+1 channel per flag" rule - it bundles TWO tensors
# (grad_rho, grad_v) under one flag, handled as a special case everywhere
# this tuple is consumed for channel counting (see in_channels below) or
# input stacking (see forward() below).
OPTIONAL_CHANNELS = ("use_grad", "use_vext", "use_rho_prev", "use_v_prev", "use_accel", "use_gamma", "use_eta")

# Non-input architecture flags that still need the same checkpoint
# save/consistency-check treatment as OPTIONAL_CHANNELS (see factory.py /
# train_phiflow.py) but must NOT be iterated by forward()'s input-channel
# stacking loop below, since they affect output interpretation, not input
# shape. Kept separate from OPTIONAL_CHANNELS for that reason.
OUTPUT_FLAGS = ("use_potential_output", "potential_smooth_sigma")
CONFIG_FLAGS = OPTIONAL_CHANNELS + OUTPUT_FLAGS

class SuperadiabaticNet(nn.Module):
    def __init__(self, channels=16, use_grad=True, use_vext=False, use_rho_prev=False, use_v_prev=False,
                 use_accel=False, use_gamma=False, use_eta=False, use_potential_output=False,
                 potential_smooth_sigma=0.0):
        super().__init__()
        # Default True: every checkpoint trained before this flag existed had
        # grad_rho/grad_v unconditionally present, so defaulting to True
        # reproduces that behavior exactly for any caller not passing use_grad.
        self.use_grad = use_grad
        self.use_vext = use_vext
        self.use_rho_prev = use_rho_prev
        self.use_v_prev = use_v_prev
        self.use_accel = use_accel
        self.use_gamma = use_gamma
        self.use_eta = use_eta
        # If True, this network's (B, X) output is interpreted as a scalar
        # superadiabatic potential field (one more spatial_gradient needed
        # to get a force - see force_factory.py:get_hybrid_force) rather
        # than a force field directly. Purely an output-interpretation
        # switch - the architecture/forward() below is identical either way.
        self.use_potential_output = use_potential_output
        # Only used when use_potential_output=True: std-dev (in grid cells)
        # of a fixed Gaussian blur applied to the raw potential field before
        # spatial_gradient - a direct, non-learned countermeasure to
        # high-frequency noise amplification (see
        # force_factory.py:_smooth_periodic). 0.0 = no smoothing (default,
        # matches original behavior exactly).
        self.potential_smooth_sigma = potential_smooth_sigma

        # Base 2 channels: [Density, Velocity]. grad(Density)/grad(Velocity)
        # are bundled under the single "use_grad" flag but count as TWO
        # channels (special-cased below, since every other OPTIONAL_CHANNELS
        # entry contributes exactly 1), plus one more for each other
        # optional channel enabled above.
        in_channels = (2 + 2 * int(self.use_grad)
                       + sum(getattr(self, flag) for flag in OPTIONAL_CHANNELS if flag != "use_grad"))

        self.net = nn.Sequential(
            # Using padding_mode='circular' mathematically guarantees periodic boundary conditions!
            nn.Conv1d(in_channels, channels, kernel_size=5, padding=2, padding_mode='circular'),
            nn.Mish(), # Mish or Swish are infinitely smooth, perfect for PDEs
            nn.Conv1d(channels, channels, kernel_size=5, padding=2, padding_mode='circular'),
            nn.Mish(),
            nn.Conv1d(channels, channels, kernel_size=5, padding=2, padding_mode='circular'),
            nn.Mish(),
            # Output is 1 channel: either F_sup directly, or (if
            # use_potential_output) a scalar potential field that
            # force_factory.py turns into F_sup via one more spatial
            # gradient - see the use_potential_output docstring above.
            nn.Conv1d(channels, 1, kernel_size=5, padding=2, padding_mode='circular')
        )

        # Initialize output weights to zero!
        # This ensures that at Epoch 0, F_sup is exactly 0.0 either way
        # (a zero potential field has a zero gradient too),
        # so the simulation starts as pure, uncorrupted equilibrium DDFT.
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=1e-2)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, rho, v, grad_rho, grad_v, v_ext=None, rho_prev=None, v_prev=None,
                accel=None, gamma=None, eta=None):
        def _squeeze(t):
            return t.squeeze(1) if t is not None and t.dim() == 3 else t

        # Ensure inputs have shape (Batch, X)
        rho, v, grad_rho, grad_v = _squeeze(rho), _squeeze(v), _squeeze(grad_rho), _squeeze(grad_v)

        channels_list = [rho, v]
        if self.use_grad:
            channels_list.append(grad_rho)
            channels_list.append(grad_v)

        provided = {"use_vext": v_ext, "use_rho_prev": rho_prev, "use_v_prev": v_prev,
                    "use_accel": accel, "use_gamma": gamma, "use_eta": eta}
        for flag in OPTIONAL_CHANNELS:
            if flag == "use_grad":
                continue  # handled above - bundles 2 tensors, not 1, see in_channels comment in __init__
            if getattr(self, flag):
                tensor = provided[flag]
                if tensor is None:
                    raise ValueError(
                        f"SuperadiabaticNet was configured with {flag}=True but no "
                        f"'{flag[len('use_'):]}' tensor was passed to forward()."
                    )
                channels_list.append(_squeeze(tensor))

        # We stack them into (Batch, in_channels, X) for Conv1d
        x = torch.stack(channels_list, dim=1)
        f_sup = self.net(x)
        return f_sup.squeeze(1) # Return shape (Batch, X)
