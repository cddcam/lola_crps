#!/usr/bin/bash

runs=(
    # --- Rayleigh Benard (Start: 5) ---
    "/mnt/ceph/users/polymathic/lola/runs/sm/yubzc3d1_rayleigh_benard_f32c16_vit_large"
    "/mnt/home/cdiaconu/polymathic_ceph/walrus_post/lola_runs/runs/sm/2zmd0rrs_rayleigh_benardFT_E100_f32c16_vit_large"
    "/mnt/ceph/users/polymathic/lola/runs/dm/e0sdjqy9_rayleigh_benard_f32c16_vit_large"
    "/mnt/home/cdiaconu/polymathic_ceph/walrus_post/lola_runs/runs/crps/V3_nkrajxou_rayleigh_benard_f32c16_vit_large_32_0.0001_0.0003_CRPS"

    # --- Euler (Start: 0) ---
    "/mnt/ceph/users/polymathic/lola/runs/sm/95zfkk6w_euler_all_f32c16_vit_large"
    "/mnt/home/cdiaconu/polymathic_ceph/walrus_post/lola_runs/runs/sm/cqbac8av_euler_allFT_E100_f32c16_vit_large"
    "/mnt/ceph/users/polymathic/lola/runs/dm/oy7s3wfq_euler_all_f32c16_vit_large"
    "/mnt/home/cdiaconu/polymathic_ceph/walrus_post/lola_runs/runs/crps/V3_bcuoxwwt_euler_all_f32c16_vit_large_32_3e-05_0.0001_CRPS"

    # --- Shear Flow (Start: 5) ---
    "/mnt/home/cdiaconu/polymathic_ceph/walrus_post/lola_runs/runs/sm/qm1x8x23_shear_flow_f32c16_vit_large"
    "/mnt/home/cdiaconu/polymathic_ceph/walrus_post/lola_runs/runs/sm/oo2gmhfd_shear_flowFT_E100_f32c16_vit_large"
    "/mnt/home/cdiaconu/polymathic_ceph/walrus_post/lola_runs/runs/crps/V3_g3j7o2vx_shear_flow_NS4_f32c16_vit_large_32_0.0001_0.003_CRPS"
)

for run in "${runs[@]}"
do
    # Assign start time based on the run name
    if [[ $run == *"rayleigh_benard"* ]]; then
        start=5
    elif [[ $run == *"shear_flow"* ]]; then
        start=5
    elif [[ $run == *"euler"* ]]; then
        start=0
    else
        # Fallback default if needed
        start=0
        echo "Warning: Unknown run type for $run, defaulting to start=0"
    fi

    # Execute the CRPS evaluation
    python experiments/eval.py start=$start seed=0 run=$run
done