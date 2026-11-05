import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import h5py
import glob
import numpy as np


def plot_movie(results_path, md_path, reference_md_path=None):
    # --- 1. Load Data ---
    file_pattern = f"{md_path}/collection.*.h5"
    md_file_list = sorted(glob.glob(file_pattern))

    with h5py.File(md_file_list[0], 'r') as f:
        V = np.array(f['V']).flatten()
        r = np.array(f['r']).flatten()
        n = np.array(f["n"]).flatten()
        vz = np.array(f["vz"]).flatten()

    ref_md_file_list = []
    if reference_md_path is not None:
        file_pattern = f"{reference_md_path}/collection.*.h5"
        ref_md_file_list = sorted(glob.glob(file_pattern))

        with h5py.File(ref_md_file_list[0], 'r') as f:
            r_2 = np.array(f['r']).flatten()
            n_2 = np.array(f["n"]).flatten()
            vz_2 = np.array(f["vz"]).flatten()

    ddft_file_list_rho = sorted(glob.glob(f"{results_path}/rho*.npy"))
    init_rho = np.load(ddft_file_list_rho[0])
    ddft_file_list_rho.pop(-1)

    ddft_file_list_v = sorted(glob.glob(f"{results_path}/v*.npy"))
    init_vz = np.load(ddft_file_list_v[0]).flatten()
    ddft_file_list_v.pop(-1)

    # --- 2. Setup Plot ---
    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.size'] = 14
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, sharex=True, figsize=(8, 8), gridspec_kw={'height_ratios':[2,1]})

    # Top Panel: DENSITY
    ax_top.set_xlim(10, 20)
    ax_top.set_ylim(0, 1.1)
    ax_top.tick_params(axis='y', labelcolor="red")
    ax_top.set_yticks(np.arange(0, 1.1, 0.1))
    ax_top.set_ylabel('Density')

    # Plot lines
    line_r_init, = ax_top.plot(r, init_rho, color='red', ls='--', alpha=.3, label='Initial')
    line_r, = ax_top.plot(r, np.load(ddft_file_list_rho[0]), lw=2, color='C3', label='DDFT')
    line_r_md, = ax_top.plot(r, init_rho, lw=2, color='C1', label='MD (NVT)')

    if reference_md_path is not None:
        line_r_md_2, = ax_top.plot(r_2, init_rho, lw=2, color='C2', label='MD (Langevin + NVE)')

    # Secondary Axis (Potential)
    ax_top2 = ax_top.twinx()
    ax_top2.set_ylabel('External potential', color="#D1D3D4")
    ax_top2.plot(r, V, label="Vext", color="#D1D3D4")
    ax_top2.tick_params(axis='y', labelcolor="#D1D3D4")
    ax_top2.set_xlim(0, 10)

    # Combine legends from both y-axes and put them ABOVE the plot
    lines_1, labels_1 = ax_top.get_legend_handles_labels()
    lines_2, labels_2 = ax_top2.get_legend_handles_labels()
    
    ax_top.legend(lines_1 + lines_2, labels_1 + labels_2, 
                  loc='lower left', 
                  bbox_to_anchor=(0., 1.02, 1., .102), # Coordinates: (x, y, width, height)
                  mode="expand", 
                  borderaxespad=0., 
                  ncol=3, 
                  fontsize=12,
                  frameon=False)

    # Bottom Panel: VELOCITY
    ax_bot.set_ylabel('Velocity ($v_z$)', color='blue')
    ax_bot.set_xlabel('Position ($z$)')
    ax_bot.axhline(0, color='black', lw=1, ls=':', alpha=0.5) # Zero-line reference
    ax_bot.tick_params(axis='y', labelcolor='blue')

    ax_bot.plot(r, init_vz, color='blue', ls='--', alpha=.3, label='Initial')
    line_vz, = ax_bot.plot(r, np.load(ddft_file_list_v[0]).flatten(), lw=2, label="DDFT", color="blue")
    line_vz_md, = ax_bot.plot(r, vz, lw=2, label="MD", color="cyan")

    line_vz_md_2 = None
    if reference_md_path is not None:
        line_vz_md_2, = ax_bot.plot(r_2, vz_2, lw=2, label="MD 2", color="dodgerblue")

    v_max = max(abs(vz.min()), abs(vz.max()), 0.25) # Avoid 0 limit
    ax_bot.set_ylim(-v_max * 2.0, v_max * 2.0)

    # Place bottom legend to the side or upper right (inside usually ok for velocity centered at 0)
    ax_bot.legend(loc='upper right', fontsize=10, framealpha=0.8)

    # Adjust layout to make room for the top legend
    plt.subplots_adjust(top=0.85, hspace=0.1)

    # --- 3. Animation ---
    N = min(len(ddft_file_list_rho), len(md_file_list))

    def update(i):
        lines = [line_r, line_vz]

        if i < len(ddft_file_list_rho):
            line_r.set_ydata(np.load(ddft_file_list_rho[i]))
            vz_ddft = np.load(ddft_file_list_v[i]).flatten()
            line_vz.set_ydata(vz_ddft)
            current_v_max = max(abs(vz_ddft.min()), abs(vz_ddft.max()))

        if i < len(md_file_list):
            with h5py.File(md_file_list[i], 'r') as f:
                n = np.array(f['n']).flatten()
                line_r_md.set_ydata(n)
                vz = np.array(f['vz']).flatten()
                line_vz_md.set_ydata(vz)
            lines.extend([line_r_md, line_vz_md])

            if reference_md_path is not None:
                with h5py.File(ref_md_file_list[i], 'r') as f:
                    n_2 = np.array(f['n']).flatten()
                    line_r_md_2.set_ydata(n_2)
                    vz_2 = np.array(f['vz']).flatten()
                    line_vz_md_2.set_ydata(vz_2)
                lines.extend([line_r_md_2, line_vz_md_2])

            # Auto-scale velocity axis if waves grow
            current_v_max = max(abs(vz.min()), abs(vz.max()), current_v_max)
            if current_v_max > ax_bot.get_ylim()[1]:
                ax_bot.set_ylim(-current_v_max*1.1, current_v_max*1.1)
                ax_bot.figure.canvas.draw()
        return lines

    anim = FuncAnimation(fig, update, frames=range(N), blit=True, repeat=False)
    FFwriter = animation.FFMpegWriter(fps=10)
    anim.save(f'{results_path}/0plot_movie.mp4', writer = FFwriter)
    #fig.savefig(f'{results_path}/0plot.png', bbox_inches='tight', pad_inches=0, transparent=True)
    fig.savefig(f'{results_path}/0plot.png', transparent=True)


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