from neural_ddft.utils.functions import plot_movie_multiple

if __name__ == "__main__":
    # output name
    output_path = "ddft_plots/comparison_solver_updated"
    output_path = "results/2026-09-28_zimmermann_hybrid_trained_01"
    # put md paths + names in here [("path/to/md1", "name1"), ...]
    md_configs = [
        #("../data/lj_data/2026-01-22_lj_nvt_01/avg_500", "NVT, Tdamp=10")
        #,("../data/lj_data/2026-02-05_lj_nve_01/avg_500", "NVE")
        #("../data/lj_data/2026-01-21_lj_langevin_01/avg_500", "MD (Langevin+NVE, Tdamp=10)")
        ("../data/lj_data/2026-01-21_lj_langevin_01/avg_500", "MD")
    ]
    # put ddft and paths + names in here [("path/to/ddft1", "name1"), ...]
    ddft_configs = [
        ("results/2026-09-28_zimmermann_hybrid_trained_01", "DDFT")
        #("results/2026-09-21_phiflow_pretrained_01", "PhiFlow Pretrained (gamma=0.1, eta=0.67)")
        #,("results/2026-09-21_phiflow_pft_01", "PhiFlow PFT (gamma=0.1, eta=0.67)")
        #,("results/2026-09-21_phiflow_zimmermann_01", "PhiFlow Zimmermann (gamma=0.1, eta=0.67)")
        #,("results/2026-09-21_pypde_pretrained_01", "PyPDE Pretrained (gamma=0.1, eta=0.67)")
        #,("../../goddard/results/2026-09-21_psmpretrained_01", "PSM Pretrained (gamma=0.1, eta=0.67)")
        #("results/2026-03-25_lj_phiflow_scale-sin+linear3_k1_T2_m1_g0.1_eta0.1_dt0.0001_c2", "PhiFlow, scaled dFdrho (gamma=0.1, eta=0.1)")
        #,("results/2026-02-25_lj_k1_T2_m1_g0.1_eta0.01_dt1e-05_c2", "Py-PDE (gamma=0.1, eta=0.01)")
        #,("results/2026-03-16_lj_closed_phiflow_k1_T2_m1_g0.1_eta0.1_dt0.0001_c2", "PhiFlow (gamma=0.1, eta=0.1)")
        #,("results/2026-03-16_lj_closed_phiflow_k1_T2_m1_g0.1_eta0.01_dt0.0001_c2", "PhiFlow, gamma=0.1, eta=0.01, dt=1e-4")
        #,("results/2026-03-16_lj_closed_phiflow_k1_T2_m1_g0.1_eta0.1_dt1e-05_c2", "PhiFlow, gamma=0.1, eta=0.1, dt=1e-5")
        #,("results/2026-03-16_lj_closed_phiflow_ktF_k1_T2_m1_g0.1_eta0.1_dt0.0001_c2", "PhiFlow (kT), gamma=0.1, eta=0.1, dt=1e-4")
        #,("results/2026-03-24_lj_phiflow_dynamic-dt_k1_T2_m1_g0.1_eta0.1_dt0.0001_c2", "PhiFlow (gamma=0.1, eta=0.1)")
        #,("../../goddard/results/2026-03-25_lj_psm_k1_T2_m1_g0.1_eta0.1_dt0.0001_c2", "PSM (gamma=0.1, eta=0.1)")
        #,("../../goddard/results/2026-03-26_lj_psm_scaled_sin+linear_k1_T2_m1_g0.1_eta0.1_dt0.0001_c2", "PSM scaled (gamma=0.1, eta=0.1)")
    ]

    plot_movie_multiple(output_path, ddft_configs, md_configs, True)
