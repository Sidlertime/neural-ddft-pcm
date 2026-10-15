from phi.torch.flow import *
import torch
import torch.nn.functional as F

from neural_ddft.physics.force_factory import FORCE_MODES

class PhiFlowDDFT:
    def __init__(self, models: dict, device, dx, config: dict):
        self.device = device
        self.dx = dx
        self.models = models

        mode = config.get("model_mode", "hybrid")
        if mode not in FORCE_MODES:
            raise ValueError(f"Unknown model_mode '{mode}'")
        self.compute_internal_force = FORCE_MODES[mode]

        self.m = float(config.get("m", 1.0))
        self.gamma = float(config.get("gamma", 2.0))
        self.eta = float(config.get("eta", 0.01))
        self.kBT = float(config.get("k", 1.0)) * float(config.get("T", 2.0))

    def step(self, rho, v, dt, V_ext, alpha=1):
        """
        Performs one integration step of the DDFT equations.
        rho: CenteredGrid
        v: StaggeredGrid
        """
        # ==========================================
        # EQUATION 1: ∂ρ/∂t = -∇(ρv)
        # ==========================================
        # Use rho.at() to interpolate density to the velocity grid faces
        rho_faces = rho.at(v)
        flux_centered = rho_faces * v

        # Calculate dynamic Artificial Diffusion to kill the wobble (Upwind stabilization)
        # This precisely targets steep gradients and shocks.
        grad_rho = field.spatial_gradient(rho)      # Lives on the faces (Staggered)
        D_art = 0.5 * self.dx * math.abs(v)         # Diffusion coeff based on local velocity
        flux_diffusive = - D_art * grad_rho         # Diffusive flux flows down the gradient
        
        # 3. Combine fluxes for a stable, mass-conserving Upwind Flux
        total_flux = flux_centered + flux_diffusive
        
        # field.divergence calculates ∇(flux) and returns it to the cell centers
        rho_next = rho - dt * field.divergence(total_flux)
        rho_next = rho_next.with_values(math.maximum(rho_next.values, 1e-6))
        
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
        f_int, _ = self.compute_internal_force(rho_next, v, self.models, self.dx, self.device)
        
        # Combine all explicit forces to update velocity
        v_next = v_advected + dt * (fric_term + visc_term + pressure_term + vext_term + f_int * alpha / self.m)
        
        return rho_next, v_next

    def save_state(self, rho, v, V_ext, current_time, results_path):
        f_total, dFdrho_grid = self.compute_internal_force(rho, v, self.models, self.dx, self.device)

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
        #np.save(f"{results_path}/f_sup{current_time:06.3f}.npy", f_sup_data)
