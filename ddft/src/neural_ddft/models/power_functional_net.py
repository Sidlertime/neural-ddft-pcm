import torch
import torch.nn as nn

from neural_ddft.models.model import OPTIONAL_CHANNELS

# PowerFunctionalNet has no output-interpretation flags (its output is
# always a true scalar, by construction - there's no vector-force/
# scalar-field ambiguity like SuperadiabaticNet's use_potential_output),
# so CONFIG_FLAGS is just the input-channel flags, reusing the same
# checkpoint consistency-check in factory.py:_load_correction_checkpoint.
CONFIG_FLAGS = OPTIONAL_CHANNELS


class PowerFunctionalNet(nn.Module):
    """
    Predicts a true scalar excess (dissipative) power functional P^ex[rho, v]
    per sample - one number, not a per-position field - analogous to how the
    frozen free-energy CNN pools density down to the scalar F_exc[rho].
    force_factory.py:get_hybrid_power_force recovers the superadiabatic
    force as f_sup = -(1/rho) * dP^ex/dv via torch.autograd.grad w.r.t. the
    velocity field - no second spatial derivative, unlike F_eq's pipeline.
    """

    def __init__(self, channels=16, use_vext=False, use_rho_prev=False, use_v_prev=False,
                 use_accel=False, use_gamma=False, use_eta=False):
        super().__init__()
        self.use_vext = use_vext
        self.use_rho_prev = use_rho_prev
        self.use_v_prev = use_v_prev
        self.use_accel = use_accel
        self.use_gamma = use_gamma
        self.use_eta = use_eta

        in_channels = 4 + sum(getattr(self, flag) for flag in OPTIONAL_CHANNELS)

        # Cascading conv+pool blocks mirror the frozen free-energy CNN's
        # pattern (circular padding for periodic BCs, Mish for the same
        # PDE-smoothness reasons as SuperadiabaticNet). AdaptiveAvgPool1d(1)
        # instead of a hardcoded downsampling kernel keeps this resolution-
        # agnostic across grids (dz32 vs dz100) by construction, unlike
        # ZimmermannNet's hardcoded-561-point window.
        self.feature_net = nn.Sequential(
            nn.Conv1d(in_channels, channels, kernel_size=5, padding=2, padding_mode='circular'),
            nn.Mish(),
            nn.AvgPool1d(kernel_size=2),
            nn.Conv1d(channels, channels, kernel_size=5, padding=2, padding_mode='circular'),
            nn.Mish(),
            nn.AvgPool1d(kernel_size=2),
            nn.Conv1d(channels, channels, kernel_size=5, padding=2, padding_mode='circular'),
            nn.Mish(),
            nn.AvgPool1d(kernel_size=2),
        )
        self.pool_head = nn.AdaptiveAvgPool1d(1)
        # No bias: f_sup = -(1/rho) * dP^ex/dv is a derivative of P^ex, and an
        # additive constant (the bias) vanishes under differentiation - it
        # would never receive a gradient from any force-based training loss
        # and would sit permanently untrained at its init value, so it's
        # omitted entirely rather than kept as dead weight.
        self.output_layer = nn.Linear(channels, 1, bias=False)

        # Small-std init (not exact zero) mirrors SuperadiabaticNet's output
        # layer exactly - P^ex starts near-zero so dP^ex/dv starts near-zero
        # too, giving the same "starts as pure equilibrium DDFT" safety
        # property every other correction network has, while avoiding a
        # perfectly symmetric zero-weight start.
        nn.init.normal_(self.output_layer.weight, mean=0.0, std=1e-2)

    def forward(self, rho, v, grad_rho, grad_v, v_ext=None, rho_prev=None, v_prev=None,
                accel=None, gamma=None, eta=None):
        def _squeeze(t):
            return t.squeeze(1) if t is not None and t.dim() == 3 else t

        rho, v, grad_rho, grad_v = _squeeze(rho), _squeeze(v), _squeeze(grad_rho), _squeeze(grad_v)

        channels_list = [rho, v, grad_rho, grad_v]
        provided = {"use_vext": v_ext, "use_rho_prev": rho_prev, "use_v_prev": v_prev,
                    "use_accel": accel, "use_gamma": gamma, "use_eta": eta}
        for flag in OPTIONAL_CHANNELS:
            if getattr(self, flag):
                tensor = provided[flag]
                if tensor is None:
                    raise ValueError(
                        f"PowerFunctionalNet was configured with {flag}=True but no "
                        f"'{flag[len('use_'):]}' tensor was passed to forward()."
                    )
                tensor = _squeeze(tensor)
                if flag == "use_accel":
                    # accel = (v - v_prev) / dt can reach magnitude ~1000+ at
                    # the project's typical dt=1e-4 (SuperadiabaticNet tolerates
                    # this fine since its path never needs a second derivative
                    # through its own forward pass). PowerFunctionalNet's force
                    # DOES require differentiating this network a second time
                    # (dP^ex/dv via autograd.grad, itself differentiated again
                    # by the outer training loss) - confirmed empirically that
                    # double-backward through Mish with unclamped accel at this
                    # magnitude overflows to NaN in the earlier conv layers.
                    # Clamping here (verified safe up to +-100, kept generous)
                    # fixes it without discarding the signal for any physically
                    # reasonable acceleration.
                    tensor = torch.clamp(tensor, -50.0, 50.0)
                channels_list.append(tensor)

        x = torch.stack(channels_list, dim=1)  # (B, in_channels, X)
        x = self.feature_net(x)                # (B, channels, X/8)
        x = self.pool_head(x).squeeze(-1)      # (B, channels)
        p_exc = self.output_layer(x).squeeze(-1)  # (B,)
        return p_exc
