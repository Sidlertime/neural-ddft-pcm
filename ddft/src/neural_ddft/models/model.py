import torch
import torch.nn as nn

# Optional input channels, in the fixed order they're stacked when enabled.
# Each flag name matches a SuperadiabaticNet constructor arg and a forward()
# keyword arg of the same name (minus "use_").
OPTIONAL_CHANNELS = ("use_vext", "use_rho_prev", "use_v_prev", "use_accel", "use_gamma", "use_eta")

class SuperadiabaticNet(nn.Module):
    def __init__(self, channels=16, use_vext=False, use_rho_prev=False, use_v_prev=False,
                 use_accel=False, use_gamma=False, use_eta=False):
        super().__init__()
        self.use_vext = use_vext
        self.use_rho_prev = use_rho_prev
        self.use_v_prev = use_v_prev
        self.use_accel = use_accel
        self.use_gamma = use_gamma
        self.use_eta = use_eta

        # Base 4 channels: [Density, Velocity, grad(Density), grad(Velocity)],
        # plus one more for each optional channel enabled above.
        in_channels = 4 + sum(getattr(self, flag) for flag in OPTIONAL_CHANNELS)

        self.net = nn.Sequential(
            # Using padding_mode='circular' mathematically guarantees periodic boundary conditions!
            nn.Conv1d(in_channels, channels, kernel_size=5, padding=2, padding_mode='circular'),
            nn.Mish(), # Mish or Swish are infinitely smooth, perfect for PDEs
            nn.Conv1d(channels, channels, kernel_size=5, padding=2, padding_mode='circular'),
            nn.Mish(),
            nn.Conv1d(channels, channels, kernel_size=5, padding=2, padding_mode='circular'),
            nn.Mish(),
            # Output is 1 channel: The Superadiabatic Force (F_sup)
            nn.Conv1d(channels, 1, kernel_size=5, padding=2, padding_mode='circular')
        )

        # Initialize output weights to zero!
        # This ensures that at Epoch 0, F_sup is exactly 0.0,
        # so the simulation starts as pure, uncorrupted equilibrium DDFT.
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=1e-2)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, rho, v, grad_rho, grad_v, v_ext=None, rho_prev=None, v_prev=None,
                accel=None, gamma=None, eta=None):
        def _squeeze(t):
            return t.squeeze(1) if t is not None and t.dim() == 3 else t

        # Ensure inputs have shape (Batch, X)
        rho, v, grad_rho, grad_v = _squeeze(rho), _squeeze(v), _squeeze(grad_rho), _squeeze(grad_v)

        channels_list = [rho, v, grad_rho, grad_v]
        provided = {"use_vext": v_ext, "use_rho_prev": rho_prev, "use_v_prev": v_prev,
                    "use_accel": accel, "use_gamma": gamma, "use_eta": eta}
        for flag in OPTIONAL_CHANNELS:
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
