from phi.torch.flow import *
import torch
import torch.nn.functional as F

from neural_ddft.physics.force_factory import FORCE_MODES

class PhiFlowDDFT:
    def __init__(self, models: dict, device, dx, config: dict):
        self.device = device
        self.dx = dx
        self.models = {}
        for name, model in models.items():
            if isinstance(model, torch.nn.Module):
                self.models[name] = model.to(self.device)
            else:
                self.models[name] = model

        mode = config.get("model_mode", "hybrid")
        if mode not in FORCE_MODES:
            raise ValueError(f"Unknown model_mode '{mode}'")
        self.compute_internal_force = FORCE_MODES[mode]

        self.m = float(config.get("m", 1.0))
        self.gamma = float(config.get("gamma", 2.0))
        self.eta = float(config.get("eta", 0.01))
        self.kBT = float(config.get("k", 1.0)) * float(config.get("T", 2.0))
        self.dt = float(config.get("dt", 1e-4))
        self.rho_floor = float(config.get("rho_floor", 1e-6))
        self.diffusion_coeff = float(config.get("diffusion_coeff", 0.5))

    def _ensure_device(self, grid):
        if grid is None:
            return None
        native_tensor = grid.values.native(grid.values.shape)
        if hasattr(native_tensor, 'device') and native_tensor.device != self.device:
            native_tensor = native_tensor.to(self.device)
            return grid.with_values(math.tensor(native_tensor, grid.values.shape))
        return grid

    def step(self, rho, v, dt=None, V_ext=None, rho_prev=None, v_prev=None, alpha=1):
        """
        Performs one integration step of the DDFT equations.
        rho: CenteredGrid
        v: StaggeredGrid
        rho_prev/v_prev: the state from one step earlier (None if unavailable,
            e.g. the very first step) - only used by sup_models configured
            with use_rho_prev/use_v_prev/use_accel.
        dt: defaults to self.dt (set from config at construction) if not given,
            so eval-side sweeps can rely on the same configured value as
            training instead of repeating it at every call site. Positional
            order is unchanged (rho, v, dt, V_ext, ...) for compatibility with
            existing positional callers (e.g. scripts/evaluation/phiflow_eval.py).
        """
        if dt is None:
            dt = self.dt
        # Guarantee all input grids reside on self.device
        rho = self._ensure_device(rho)
        v = self._ensure_device(v)
        V_ext = self._ensure_device(V_ext)

        # ==========================================
        # EQUATION 1: ∂ρ/∂t = -∇(ρv)
        # ==========================================
        # Use rho.at() to interpolate density to the velocity grid faces
        rho_faces = rho.at(v)
        flux_centered = rho_faces * v

        # Calculate dynamic Artificial Diffusion to kill the wobble (Upwind stabilization)
        # This precisely targets steep gradients and shocks.
        grad_rho = field.spatial_gradient(rho)      # Lives on the faces (Staggered)
        D_art = self.diffusion_coeff * self.dx * math.abs(v)  # Diffusion coeff based on local velocity
        flux_diffusive = - D_art * grad_rho         # Diffusive flux flows down the gradient
        
        # 3. Combine fluxes for a stable, mass-conserving Upwind Flux
        total_flux = flux_centered + flux_diffusive
        
        # field.divergence calculates ∇(flux) and returns it to the cell centers
        rho_next = rho - dt * field.divergence(total_flux)
        rho_next = rho_next.with_values(math.maximum(rho_next.values, self.rho_floor))
        
        # ==========================================
        # EQUATION 2: ∂v/∂t = ... (Velocity Balance)
        # ==========================================
        
        # Advection: -v∇v 
        v_advected = advect.semi_lagrangian(v, v, dt)
        
        # Resample updated density to faces for the remaining division terms
        rho_faces_next = rho_next.at(v)
        
        # Friction: -γv
        fric_term = -self.gamma * v
        
        # Viscous Diffusion: + (η/(mρ)) ∇²v
        # Use field.laplace for physical grids
        visc_term = (self.eta / (self.m * rho_faces_next)) * field.laplace(v)
        
        # Ideal Gas Pressure: - (k_B T / (mρ)) ∇ρ
        # Use field.spatial_gradient to map from Centers to Faces
        pressure_term = - (self.kBT / (self.m * rho_faces_next)) * field.spatial_gradient(rho_next)
        
        # External Potential: - (1/m) ∇V_ext
        vext_term = - (1.0 / self.m) * field.spatial_gradient(V_ext)
        
        # Internal Force based on Excess Free Energy: - (1/m) ∇(δF_exc/δρ)
        f_int, _, _ = self.compute_internal_force(
            rho_next, v, self.models, self.dx, self.device,
            V_ext_grid=V_ext, rho_prev_grid=rho_prev, v_prev_grid=v_prev, dt=dt,
            gamma=self.gamma, eta=self.eta,
        )
        
        # Combine all explicit forces to update velocity
        v_next = v_advected + dt * (fric_term + visc_term + pressure_term + vext_term + f_int * alpha / self.m)
        
        return rho_next, v_next

    def save_state(self, rho, v, V_ext, current_time, results_path):
        # No rho_prev/v_prev available at a standalone save point - get_hybrid_force
        # falls back to rho_prev=rho/v_prev=v (zero acceleration), which is fine for
        # a diagnostic snapshot (not part of the actual integration rollout). Any
        # nonzero dt is safe to pass here since that fallback makes the
        # accel numerator (v - v_prev) exactly zero regardless of dt's value -
        # dt=1.0 is just a placeholder to satisfy use_accel's "dt is required" check.
        f_total, dFdrho_grid, force_components = self.compute_internal_force(
            rho, v, self.models, self.dx, self.device,
            V_ext_grid=V_ext, dt=1.0, gamma=self.gamma, eta=self.eta,
        )

        rho_data = rho.values.native('batch,x')[0].cpu().detach().numpy()
        v_data = v.values.vector['x'].native('batch,x')[0].cpu().detach().numpy()

        f_total_data = f_total.values.vector['x'].native('batch,x')[0].cpu().detach().numpy()
        dFdrho_data = dFdrho_grid.values.native('batch,x')[0].cpu().detach().numpy()

        if hasattr(V_ext.values, 'native'):
            V_ext_np = V_ext.values.native('batch,x')[0].cpu().detach().numpy()
        else:
            V_ext_np = V_ext.values.numpy('batch,x')[0]

        np.save(f"{results_path}/rho{current_time:06.3f}.npy", rho_data)
        np.save(f"{results_path}/v{current_time:06.3f}.npy", v_data)
        np.save(f"{results_path}/V_ext{current_time:06.3f}.npy", V_ext_np)
        np.save(f"{results_path}/dFdrho{current_time:06.3f}.npy", dFdrho_data)
        np.save(f"{results_path}/dFdrho_mod{current_time:06.3f}.npy", f_total_data)

        # Persist each force component (f_eq, and f_sup/f_zimmermann if present)
        # separately so base/added/total can be plotted side by side.
        force_ratio = None
        f_eq_norm = None
        for comp_name, comp_grid in force_components.items():
            comp_data = comp_grid.values.vector['x'].native('batch,x')[0].cpu().detach().numpy()
            np.save(f"{results_path}/{comp_name}{current_time:06.3f}.npy", comp_data)
            if comp_name == "f_eq":
                f_eq_norm = np.linalg.norm(comp_data)
            elif comp_name in ("f_sup", "f_zimmermann", "f_power"):
                force_ratio = np.linalg.norm(comp_data)

        if force_ratio is not None and f_eq_norm is not None:
            force_ratio = float(force_ratio / (f_eq_norm + 1e-8))

        return {"force_ratio": force_ratio}
