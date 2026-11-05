import os
import glob
import numpy as np
import h5py


def load_npy_series(directory, prefix="rho"):
    """
    Loads series of numpy arrays saved as {directory}/{prefix}{time:06.3f}.npy
    Returns sorted list of (time, array).
    """
    files = sorted(glob.glob(os.path.join(directory, f"{prefix}*.npy")))
    result = []
    for f in files:
        basename = os.path.basename(f)
        time_str = basename[len(prefix):-4]
        try:
            t = float(time_str)
            data = np.load(f)
            result.append((t, data))
        except ValueError:
            continue
    result.sort(key=lambda item: item[0])
    return result


def load_md_series(md_path):
    """
    Loads MD files matching collection.*.h5.
    Returns times, densities (n), velocities (vz), and grid r.
    """
    files = sorted(glob.glob(os.path.join(md_path, "collection.*.h5")))
    times = []
    rhos = []
    vs = []
    r_grid = None
    for idx, f in enumerate(files):
        with h5py.File(f, "r") as fp:
            n = np.array(fp["n"]).flatten()
            vz = np.array(fp["vz"]).flatten() if "vz" in fp else None
            if r_grid is None and "r" in fp:
                r_grid = np.array(fp["r"]).flatten()
            rhos.append(n)
            if vz is not None:
                vs.append(vz)
            times.append(idx)
    return np.array(times), np.array(rhos), (np.array(vs) if vs else None), r_grid


def compute_metrics(pred, target):
    """
    Computes MAE, MSE, and Linf (max absolute error) between two aligned 1D or 2D arrays.
    """
    diff = np.abs(pred - target)
    mae = float(np.mean(diff))
    mse = float(np.mean(diff ** 2))
    linf = float(np.max(diff))
    return {"mae": mae, "mse": mse, "linf": linf}


def evaluate_single_prediction(pred_dir, ref_dir_or_md, is_ref_md=False, max_steps=None):
    """
    Evaluates one prediction directory against reference data.
    """
    pred_rho = load_npy_series(pred_dir, prefix="rho")
    pred_v = load_npy_series(pred_dir, prefix="v")

    if not pred_rho:
        raise FileNotFoundError(f"No rho*.npy files found in pred_dir: {pred_dir}")

    r_grid = None
    if is_ref_md:
        _, ref_rho_arr, ref_v_arr, r_grid = load_md_series(ref_dir_or_md)
        if ref_rho_arr is None or len(ref_rho_arr) == 0:
            raise FileNotFoundError(f"No MD collection files found in ref path: {ref_dir_or_md}")
        n_frames = min(len(pred_rho), len(ref_rho_arr))
        if max_steps is not None:
            n_frames = min(n_frames, max_steps)

        times = np.array([pred_rho[i][0] for i in range(n_frames)])
        p_rho = np.array([pred_rho[i][1] for i in range(n_frames)])
        t_rho = ref_rho_arr[:n_frames]

        has_v = (len(pred_v) >= n_frames) and (ref_v_arr is not None and len(ref_v_arr) >= n_frames)
        p_v = np.array([pred_v[i][1] for i in range(n_frames)]) if has_v else None
        t_v = ref_v_arr[:n_frames] if has_v else None
    else:
        ref_rho = load_npy_series(ref_dir_or_md, prefix="rho")
        ref_v = load_npy_series(ref_dir_or_md, prefix="v")

        if not ref_rho:
            raise FileNotFoundError(f"No rho*.npy files found in ref_dir: {ref_dir_or_md}")

        time_map = {round(t_ref, 4): d for t_ref, d in ref_rho}
        matched_times = []
        p_rho_list = []
        t_rho_list = []
        for t_p, data_p in pred_rho:
            key = round(t_p, 4)
            if key in time_map:
                matched_times.append(t_p)
                p_rho_list.append(data_p)
                t_rho_list.append(time_map[key])

        if max_steps is not None:
            matched_times = matched_times[:max_steps]
            p_rho_list = p_rho_list[:max_steps]
            t_rho_list = t_rho_list[:max_steps]

        times = np.array(matched_times)
        p_rho = np.array(p_rho_list)
        t_rho = np.array(t_rho_list)

        has_v = bool(ref_v and pred_v)
        if has_v:
            time_map_v = {round(t, 4): d for t, d in ref_v}
            p_v_list = []
            t_v_list = []
            for t_p, data_p in pred_v:
                key = round(t_p, 4)
                if key in time_map_v and (max_steps is None or len(p_v_list) < max_steps):
                    p_v_list.append(data_p)
                    t_v_list.append(time_map_v[key])
            p_v = np.array(p_v_list)
            t_v = np.array(t_v_list)
        else:
            p_v = None
            t_v = None

    if len(p_rho) == 0:
        raise ValueError("No matching time points found between prediction and reference.")

    per_step_rho = [compute_metrics(p_rho[i], t_rho[i]) for i in range(len(p_rho))]
    overall_rho = compute_metrics(p_rho, t_rho)

    report = {
        "pred_path": pred_dir,
        "ref_path": ref_dir_or_md,
        "n_frames_evaluated": len(p_rho),
        "times": times,
        "r_grid": r_grid,
        "p_rho": p_rho,
        "t_rho": t_rho,
        "p_v": p_v,
        "t_v": t_v,
        "per_step_rho": per_step_rho,
        "density_metrics": {
            "mean_mae": float(np.mean([m["mae"] for m in per_step_rho])),
            "mean_mse": float(np.mean([m["mse"] for m in per_step_rho])),
            "max_linf": float(np.max([m["linf"] for m in per_step_rho])),
            "overall_mse": overall_rho["mse"],
            "overall_linf": overall_rho["linf"],
        }
    }

    if has_v and p_v is not None and t_v is not None and len(p_v) == len(t_v) and len(p_v) > 0:
        per_step_v = [compute_metrics(p_v[i], t_v[i]) for i in range(len(p_v))]
        overall_v = compute_metrics(p_v, t_v)
        report["per_step_v"] = per_step_v
        report["velocity_metrics"] = {
            "mean_mae": float(np.mean([m["mae"] for m in per_step_v])),
            "mean_mse": float(np.mean([m["mse"] for m in per_step_v])),
            "max_linf": float(np.max([m["linf"] for m in per_step_v])),
            "overall_mse": overall_v["mse"],
            "overall_linf": overall_v["linf"],
        }

    return report


def summarize_run(pred_dir, ref_dir_or_md=None, is_ref_md=False, max_steps=None):
    """
    Thin wrapper returning only the JSON-serializable subset of
    evaluate_single_prediction's report (density_metrics/velocity_metrics),
    or None if no reference is available to compare against.
    """
    if ref_dir_or_md is None:
        return None
    rep = evaluate_single_prediction(pred_dir, ref_dir_or_md, is_ref_md=is_ref_md, max_steps=max_steps)
    return {"density_metrics": rep["density_metrics"], "velocity_metrics": rep.get("velocity_metrics")}
