#!/bin/bash
# run_eval_suite.sh - Evaluate multiple DDFT models/runs
# MUST BE RUN FROM THE `ddft` DIRECTORY

set -e

# --- Placeholder Variables (Standard Parameters for future use) ---
BASE_MODEL_PATH="models/c2_dz32.ckpt"
SUP_MODEL_PATH=""           # e.g., "models/sup_hybrid.ckpt"
ZIMMERMANN_MODEL_PATH=""    # e.g., "models/sup_zimmermann.ckpt"
REF_MD_PATH=""
K_VAL="1.0"
T_VAL="2.0"
M_VAL="1.0"
DT_VAL="1e-4"
# ----------------------------------------------------------------

# --- Active Parameters ---
GAMMA="0.1"
ETA="0.6944"

# Target Data Arrays
MD_DIRECTORIES=(
    "../data/lj_data/2026-10-23_ref_sim_3a_langevin_01/avg_2000" 
    "../data/lj_data/2026-10-24_ref_sim_3b_langevin_01/avg_2000" 
    "../data/lj_data/2026-10-24_ref_sim_3c_langevin_01/avg_2000" 
    "../data/lj_data/2026-10-24_ref_sim_3d_langevin_01/avg_2000"
)
RUN_IDS=("ref_3a" "ref_3b" "ref_3c" "ref_3d")
#MODES=("equilibrium" "hybrid" "zimmermann")
MODES=("equilibrium")

echo "Starting evaluation suite from $(pwd)..."

for mode in "${MODES[@]}"; do
    for i in "${!MD_DIRECTORIES[@]}"; do         
        dir="${MD_DIRECTORIES[$i]}"
        current_run_id="${RUN_IDS[$i]}"

        if [ -d "$dir" ]; then
            echo "========================================================="
            echo "Evaluating Mode: $mode | Run ID: $current_run_id"
            echo "Data: $dir"
            echo "========================================================="
            
            # Execute the evaluation script
            python scripts/evaluation/phiflow_eval.py \
                --run_id "$current_run_id" \
                --model_mode "$mode" \
                --md_path "$dir" \
                --gamma "$GAMMA" \
                --eta "$ETA" \
                --base_model_path "$BASE_MODEL_PATH"
                
                # Uncomment and add these when you want to test specific supervised nets:
                # --sup_model_path "$SUP_MODEL_PATH" \
                # --zimmermann_model_path "$ZIMMERMANN_MODEL_PATH" \
                
        else
            echo "Skipping: $DATA_DIR not found."
        fi
    done
done

echo "Evaluation suite completed."