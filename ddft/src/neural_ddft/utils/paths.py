"""
Central locations of data, results and the sibling project repositories.

Every location has a default relative to this repository and can be overridden
with an environment variable of the same name:

    REPO_ROOT      neural-ddft-pcm/                      (repository root)
    DDFT_ROOT      neural-ddft-pcm/ddft/                 (working directory of all scripts)
    DATA_ROOT      neural-ddft-pcm/data/lj_data/         (Langevin MD reference data)
    RESULTS_ROOT   neural-ddft-pcm/ddft/results/
    GODDARD_ROOT   ../goddard/                           (github.com/Sidlertime/goddard-ddft)
    MDEXT_ROOT     ../mdext-ddft/                        (github.com/Sidlertime/mdext-ddft)

MD data layout (written by mdext-ddft, see its README):

    DATA_ROOT/ref_<potential>[_gamma<digits>]_langevin_01/avg_<trials>/collection.NNN.h5
    (e.g. ref_3a_langevin_01 for gamma = 0.1, ref_3a_gamma005_langevin_01 for gamma = 0.05)
"""
import glob
import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

DDFT_ROOT = os.environ.get("DDFT_ROOT", os.path.abspath(os.path.join(_THIS_DIR, "..", "..", "..")))
REPO_ROOT = os.environ.get("REPO_ROOT", os.path.dirname(DDFT_ROOT))
DATA_ROOT = os.environ.get("DATA_ROOT", os.path.join(REPO_ROOT, "data", "lj_data"))
RESULTS_ROOT = os.environ.get("RESULTS_ROOT", os.path.join(DDFT_ROOT, "results"))
GODDARD_ROOT = os.environ.get("GODDARD_ROOT", os.path.join(os.path.dirname(REPO_ROOT), "goddard"))
MDEXT_ROOT = os.environ.get("MDEXT_ROOT", os.path.join(os.path.dirname(REPO_ROOT), "mdext-ddft"))

# Friction of the reference data. gamma = 1/Tdamp of the Langevin thermostat.
DEFAULT_GAMMA = 0.1


def gamma_tag(gamma):
    """Folder tag of a friction value: 0.1 -> "" (default), 0.05 -> "_gamma005", 0.5 -> "_gamma05"."""
    if abs(gamma - DEFAULT_GAMMA) < 1e-12:
        return ""
    return "_gamma" + f"{gamma:g}".replace(".", "")


def md_dir(potential, gamma=DEFAULT_GAMMA, data_root=None):
    """
    Folder of the averaged MD frames of one reference dataset, e.g.
    md_dir("3a") -> DATA_ROOT/ref_3a_langevin_01/avg_2000.

    Raises FileNotFoundError with a hint how to create the data if it is missing.
    """
    base = os.path.join(data_root or DATA_ROOT, f"ref_{potential}{gamma_tag(gamma)}_langevin_01")
    candidates = sorted(glob.glob(os.path.join(base, "avg_*")))
    if not candidates:
        raise FileNotFoundError(
            f"MD data for potential '{potential}' (gamma={gamma}) not found in {base}/avg_*. "
            "Generate it with mdext-ddft (examples/lj/run_collection.sh) and link it there, "
            "see the README.")
    # several averages of the same dataset: use the one with the most trials
    return max(candidates, key=lambda p: int(os.path.basename(p).split("_")[-1]))


def resolve(path, root=None):
    """Absolute path; relative paths are taken relative to `root` (default DDFT_ROOT)."""
    if path is None or os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(root or DDFT_ROOT, path))
