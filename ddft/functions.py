import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import h5py
import glob
import numpy as np
import os

class SimData:
    """Helper class to lazily load and store simulation frame paths and plotting objects."""
    def __init__(self, path, name, color, is_md=False):
        self.path = path
        self.name = name
        self.color = color
        self.is_md = is_md
        
        # File paths
        self.rho_files = []
        self.v_files = []
        self.df_files = []
        self.md_files = []
        self.has_v = False
        
        # Line objects for animation
        self.line_rho = None
        self.line_v = None
        self.line_df = None
        self.line_df_mod = None

        if is_md:
            self.md_files = sorted(glob.glob(f"{path}/collection.*.h5"))
            if self.md_files:
                with h5py.File(self.md_files[0], 'r') as f:
                    if 'vz' in f:
                        self.has_v = True
        else:
            self.rho_files = sorted(glob.glob(f"{path}/rho*.npy"))
            self.v_files = sorted(glob.glob(f"{path}/v*.npy"))
            self.df_files = sorted(glob.glob(f"{path}/dFdrho*.npy"))
            self.df_mod_files = sorted(glob.glob(f"{path}/dFdrho_mod*.npy"))
            self.has_v = len(self.v_files) > 0

    @property
    def frames(self):
        if self.is_md:
            return len(self.md_files)
        return len(self.rho_files)


def _get_static_reference(ddfts, mds):
    """Extracts the spatial grid (r), initial density, and external potential once."""
    r, init_rho, V = None, None, None
    
    # Prefer MD for the spatial grid 'r'
    if mds and mds[0].frames > 0:
        with h5py.File(mds[0].md_files[0], 'r') as f:
            r = np.array(f['r']).flatten()
            init_rho = np.array(f['n']).flatten()
            V = np.array(f.get('V', f.get('V_ext'))).flatten()
    elif ddfts and ddfts[0].frames > 0:
        init_rho = np.load(ddfts[0].rho_files[0]).flatten()
        # Fallback for r if no MD is provided (assuming domain 0 to 10)
        r = np.linspace(0, 10, len(init_rho))
        # Find V_ext file
        v_ext_files = sorted(glob.glob(f"{ddfts[0].path}/V_ext*.npy"))
        if v_ext_files:
            V = np.load(v_ext_files[0]).flatten()
        else:
            V = np.zeros_like(init_rho)
            
    return r, init_rho, V


def plot_movie_multiple(output_path, ddft_configs=None, md_configs=None, plot_dfdrho=False):
    """
    Plots an arbitrary number of DDFT and MD simulations.
    
    Args:
        output_path (str): Directory to save the output MP4 and PNG.
        ddft_configs (list of tuples): [(path, label), (path, label), ...]
        md_configs (list of tuples): [(path, label), (path, label), ...]
        plot_dfdrho (bool): If True, adds a 3rd subplot for dFdrho.
    """
    ddft_configs = ddft_configs or []
    md_configs = md_configs or []
    
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_idx = 0
    
    # 1. Initialize data handlers
    sims = []
    for path, name in ddft_configs:
        sims.append(SimData(path, name, colors[color_idx % len(colors)], is_md=False))
        color_idx += 1
        
    for path, name in md_configs:
        sims.append(SimData(path, name, colors[color_idx % len(colors)], is_md=True))
        color_idx += 1
        
    if not sims:
        print("No simulations provided to plot.")
        return

    # 2. Extract static baseline data
    ddfts = [s for s in sims if not s.is_md]
    mds = [s for s in sims if s.is_md]
    r, init_rho, V = _get_static_reference(ddfts, mds)
    
    max_frames = min([s.frames for s in sims if s.frames > 0])

    # 3. Setup Plot Layout
    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.size'] = 14
    
    panels = 3 if plot_dfdrho else 2
    ratios = [2, 1, 1] if plot_dfdrho else [2, 1]
    fig, axes = plt.subplots(panels, 1, sharex=True, figsize=(8, 4 * panels), gridspec_kw={'height_ratios': ratios})
    
    ax_top = axes[0]
    ax_bot = axes[1]
    ax_df = axes[2] if plot_dfdrho else None

    # --- TOP PANEL: Density ---
    ax_top.set_xlim(r[0], r[-1])
    ax_top.set_ylim(0, max(1.1, init_rho.max() * 1.2))
    ax_top.set_ylabel('Density', color='C3')
    ax_top.tick_params(axis='y', labelcolor='C3')
    
    # Static Initial Density
    ax_top.plot(r, init_rho, color='gray', ls='--', alpha=0.5, label='Initial')
    
    # Static External Potential (Secondary Axis)
    ax_top2 = ax_top.twinx()
    ax_top2.set_ylabel('External potential', color="#D1D3D4")
    ax_top2.plot(r, V, label="Vext", color="#D1D3D4")
    ax_top2.tick_params(axis='y', labelcolor="#D1D3D4")

    # --- BOTTOM PANEL: Velocity ---
    ax_bot.set_ylabel('Velocity ($v_z$)', color='blue')
    if not plot_dfdrho:
        ax_bot.set_xlabel('Position ($z$)')
    ax_bot.axhline(0, color='black', lw=1, ls=':', alpha=0.5)
    ax_bot.tick_params(axis='y', labelcolor='blue')
    ax_bot.set_ylim(-0.5, 0.5) # Initial limits, will autoscale

    # --- OPTIONAL 3RD PANEL: dFdrho ---
    if plot_dfdrho:
        ax_df.set_ylabel('$\\delta F_{exc} / \\delta \\rho$', color='purple')
        ax_df.set_xlabel('Position ($z$)')
        ax_df.axhline(0, color='black', lw=1, ls=':', alpha=0.5)
        ax_df.tick_params(axis='y', labelcolor='purple')
        ax_df.set_ylim(-1.0, 1.0) # Will autoscale

    # 4. Initialize dynamic lines for each simulation
    for sim in sims:
        # Density line
        sim.line_rho, = ax_top.plot([], [], lw=2, color=sim.color, label=sim.name)
        
        # Velocity line
        if sim.has_v:
            sim.line_v, = ax_bot.plot([], [], lw=2, color=sim.color, label=sim.name)
            
        # dFdrho line (DDFT only)
        if plot_dfdrho and not sim.is_md and len(sim.df_files) > 0:
            sim.line_df, = ax_df.plot([], [], lw=2, color=sim.color, label=sim.name)
            if len(sim.df_mod_files) > 0:
                sim.line_df_mod, = ax_df.plot([], [], lw=2, color=sim.color, ls='--', label=f'{sim.name} (scaled)')

    # Clean up legends
    lines_1, labels_1 = ax_top.get_legend_handles_labels()
    lines_2, labels_2 = ax_top2.get_legend_handles_labels()
    ax_top.legend(lines_1 + lines_2, labels_1 + labels_2, 
                  loc='lower left', 
                  bbox_to_anchor=(0.0, 1.05), # Pins it just above the top left corner
                  borderaxespad=0., 
                  ncol=1,                     # 1 column forces every long name onto its own line
                  fontsize=10,                # Slightly smaller font to fit configuration details
                  frameon=False)

    # 5. Animation Update Function
    def update(i):
        drawn_artists = []
        v_min_current, v_max_current = ax_bot.get_ylim()
        df_min_current, df_max_current = (-1.0, 1.0) if not plot_dfdrho else ax_df.get_ylim()
        
        needs_v_autoscale = False
        needs_df_autoscale = False

        for sim in sims:
            if i >= sim.frames:
                continue
                
            if sim.is_md:
                with h5py.File(sim.md_files[i], 'r') as f:
                    # Update Density
                    rho_data = np.array(f['n']).flatten()
                    sim.line_rho.set_data(r, rho_data)
                    drawn_artists.append(sim.line_rho)
                    
                    # Update Velocity if it exists
                    if sim.has_v:
                        v_data = np.array(f['vz']).flatten()
                        sim.line_v.set_data(r, v_data)
                        drawn_artists.append(sim.line_v)
                        # Check bounds
                        v_max_val = np.max(np.abs(v_data))
                        if v_max_val > v_max_current:
                            v_max_current = v_max_val
                            needs_v_autoscale = True
            else:
                # Update DDFT Density
                rho_data = np.load(sim.rho_files[i]).flatten()
                sim.line_rho.set_data(r, rho_data)
                drawn_artists.append(sim.line_rho)
                
                # Update DDFT Velocity
                if sim.has_v:
                    v_data = np.load(sim.v_files[i]).flatten()
                    sim.line_v.set_data(r, v_data)
                    drawn_artists.append(sim.line_v)
                    v_max_val = np.max(np.abs(v_data))
                    if v_max_val > v_max_current:
                        v_max_current = v_max_val
                        needs_v_autoscale = True
                
                # Update DDFT dFdrho
                if plot_dfdrho:
                    if sim.line_df is not None:
                        df_data = np.load(sim.df_files[i]).flatten()
                        sim.line_df.set_data(r, df_data)
                        drawn_artists.append(sim.line_df)
                        
                        df_max_val = np.max(np.abs(df_data))
                        if df_max_val > df_max_current:
                            df_max_current = df_max_val
                            needs_df_autoscale = True
                    
                    if sim.line_df_mod is not None and i < len(sim.df_mod_files):
                        df_mod_data = np.load(sim.df_mod_files[i]).flatten()
                        sim.line_df_mod.set_data(r, df_mod_data)
                        drawn_artists.append(sim.line_df_mod)
                        df_max_val = np.max(np.abs(df_mod_data))
                        if df_max_val > df_max_current:
                            df_max_current = df_max_val
                            needs_df_autoscale = True

        # Apply Autoscaling if boundaries were broken
        if needs_v_autoscale:
            ax_bot.set_ylim(-v_max_current * 1.2, v_max_current * 1.2)
            ax_bot.figure.canvas.draw_idle()
            
        if needs_df_autoscale and plot_dfdrho:
            ax_df.set_ylim(-df_max_current * 1.2, df_max_current * 1.2)
            ax_df.figure.canvas.draw_idle()

        return drawn_artists

    # 6. Save Animation and Final Frame
    plt.subplots_adjust(top=0.75, hspace=0.1)
    anim = FuncAnimation(fig, update, frames=range(max_frames), blit=True, repeat=False)
    
    os.makedirs(output_path, exist_ok=True)
    anim.save(f'{output_path}/0plot_movie.mp4', writer=FFMpegWriter(fps=10))
    fig.savefig(f'{output_path}/0plot.png', transparent=True, bbox_inches='tight')
    plt.close(fig)


def plot_movie(results_path, md_path, reference_md_path=None, plot_dfdrho=False):
    """
    Backwards-compatible wrapper for single DDFT and 1-2 MD comparisons.
    """
    ddft_configs = [(results_path, 'DDFT')]
    
    md_configs = [(md_path, 'MD')]
    if reference_md_path:
        md_configs.append((reference_md_path, 'MD 2'))
        
    plot_movie_multiple(
        output_path=results_path,
        ddft_configs=ddft_configs,
        md_configs=md_configs,
        plot_dfdrho=plot_dfdrho
    )

# Artifact from previous script
def plot_movie_open(results_path, md_path, init_rho):
    file_pattern = f"{md_path}/collection.*.h5"
    md_file_list = sorted(glob.glob(file_pattern))

    with h5py.File(md_file_list[0], 'r') as f:
        init_rho = np.array(f['n']).flatten()
        V = np.array(f['V']).flatten()
        r = np.array(f['r']).flatten()

    ddft_file_list_rho = sorted(glob.glob(f"{results_path}/rho*.npy"))

    fig, ax_top = plt.subplots()
    ax_top.set_xlim(0, 10)
    ax_top.set_ylim(0, 2)
    ax_top.set_yticks(np.arange(0, 1.1, 0.1))
    ax_top.plot(r, init_rho, color='red', ls='--', alpha=.3, label='Initial')
    line_r, = ax_top.plot(r, np.load(ddft_file_list_rho[0]), lw=2, color='C3', label='DDFT')

    ax_top2 = ax_top.twinx()
    ax_top2.set_ylabel('External potential', color="#D1D3D4")
    ax_top2.plot(r, V, label="Vext", color="#D1D3D4")
    ax_top2.tick_params(axis='y', labelcolor="#D1D3D4")
    ax_top2.set_xlim(0, 10)

    line_r_md, = ax_top.plot(r, init_rho, lw=2, color='C1', label='MD')
    ax_top.legend()

    N = min(len(ddft_file_list_rho), len(md_file_list))

    def update(i):
        if i < len(ddft_file_list_rho):
            line_r.set_ydata(np.load(ddft_file_list_rho[i]))
        if i < len(md_file_list):
            with h5py.File(md_file_list[i], 'r') as f:
                n = np.array(f['n']).flatten()
                line_r_md.set_ydata(n)
        return line_r, line_r_md,

    anim = FuncAnimation(fig, update, frames=range(N), blit=True, repeat=False)
    FFwriter = animation.FFMpegWriter(fps=10)
    anim.save(f'{results_path}/0plot_movie.mp4', writer = FFwriter)
    fig.savefig(f'{results_path}/0plot.png')
