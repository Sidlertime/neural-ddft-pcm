import os
from types import SimpleNamespace
import numpy as np
import pytest
from neural_ddft.data.dataset import MDWindowDataset, MultiPotentialMDDataset

GAUSSIAN_PATH = "../data/lj_data/ref_gaussian_langevin_01/avg_500"
REF_3A_PATH = "../data/lj_data/ref_3a_langevin_01/avg_2000"
REF_3B_PATH = "../data/lj_data/ref_3b_langevin_01/avg_2000"

_data_available = os.path.isdir(GAUSSIAN_PATH) and os.path.isdir(REF_3A_PATH) and os.path.isdir(REF_3B_PATH)
requires_md_data = pytest.mark.skipif(not _data_available, reason="Reference MD data directories not found")


@requires_md_data
def test_multi_potential_combines_lengths_and_tags_potential_id():
    print("\n[DEBUG] Starting test_multi_potential_combines_lengths_and_tags_potential_id...")
    window_size = 5
    single_gaussian = MDWindowDataset(GAUSSIAN_PATH, window_size=window_size, stride=1)
    single_3a = MDWindowDataset(REF_3A_PATH, window_size=window_size, stride=1)
    single_3b = MDWindowDataset(REF_3B_PATH, window_size=window_size, stride=1)

    combined = MultiPotentialMDDataset(
        [GAUSSIAN_PATH, REF_3A_PATH, REF_3B_PATH], window_size=window_size, stride=1
    )

    assert len(combined) == len(single_gaussian) + len(single_3a) + len(single_3b)
    assert combined.V_ext.shape == (3, combined.domain_resolution)

    # First sample must come from potential 0 (gaussian), last from potential 2 (3b)
    assert combined[0]['potential_id'] == 0
    assert combined[len(combined) - 1]['potential_id'] == 2

    # Spot-check a sample squarely inside the second potential's range
    mid_3a_idx = len(single_gaussian) + len(single_3a) // 2
    assert combined[mid_3a_idx]['potential_id'] == 1
    print("[DEBUG] test_multi_potential_combines_lengths_and_tags_potential_id passed successfully.")


@requires_md_data
def test_phase_chunk_indices_stays_within_one_potential():
    print("\n[DEBUG] Starting test_phase_chunk_indices_stays_within_one_potential...")
    window_size = 5
    combined = MultiPotentialMDDataset(
        [GAUSSIAN_PATH, REF_3A_PATH], window_size=window_size, stride=1
    )

    indices = combined.phase_chunk_indices(chunk_fraction=0.15, rng=np.random.RandomState(0))

    # Every chunk's sampled indices must all map back to a single potential_id
    offset = 0
    for length in combined._lengths:
        chunk_here = [i for i in indices if offset <= i < offset + length]
        potential_ids = {combined[i]['potential_id'] for i in chunk_here}
        assert len(potential_ids) <= 1, "A per-phase chunk must not straddle two potentials"
        offset += length
    print("[DEBUG] test_phase_chunk_indices_stays_within_one_potential passed successfully.")


@requires_md_data
def test_phase_chunk_indices_start_fraction_is_fixed_and_deterministic():
    print("\n[DEBUG] Starting test_phase_chunk_indices_start_fraction_is_fixed_and_deterministic...")
    window_size = 5
    combined = MultiPotentialMDDataset(
        [GAUSSIAN_PATH, REF_3A_PATH], window_size=window_size, stride=1
    )

    # Same start_fraction, called twice with different (or no) rng - must be identical,
    # since start_fraction bypasses the random draw entirely.
    indices_a = combined.phase_chunk_indices(chunk_fraction=0.15, start_fraction=0.4)
    indices_b = combined.phase_chunk_indices(chunk_fraction=0.15, start_fraction=0.4, rng=np.random.RandomState(123))
    assert indices_a == indices_b

    # Each potential's chunk must actually start at ~40% of its own length (clamped to fit).
    offset = 0
    for length in combined._lengths:
        chunk_here = sorted(i - offset for i in indices_a if offset <= i < offset + length)
        chunk_size = max(1, int(0.15 * length))
        expected_start = min(int(0.4 * length), max(0, length - chunk_size))
        assert chunk_here[0] == expected_start
        offset += length

    # A different start_fraction must move the chunk.
    indices_c = combined.phase_chunk_indices(chunk_fraction=0.15, start_fraction=0.1)
    assert indices_a != indices_c
    print("[DEBUG] test_phase_chunk_indices_start_fraction_is_fixed_and_deterministic passed successfully.")


@requires_md_data
def test_phase_chunk_indices_start_min_fraction_bounds_random_draw():
    print("\n[DEBUG] Starting test_phase_chunk_indices_start_min_fraction_bounds_random_draw...")
    window_size = 5
    combined = MultiPotentialMDDataset(
        [GAUSSIAN_PATH, REF_3A_PATH], window_size=window_size, stride=1
    )

    # With start_min_fraction=0.5, every potential's chunk must start at or
    # past 50% of its own trajectory length, across many random draws.
    offset_lengths = []
    offset = 0
    for length in combined._lengths:
        offset_lengths.append((offset, length))
        offset += length

    for seed in range(20):
        indices = combined.phase_chunk_indices(
            chunk_fraction=0.15, start_min_fraction=0.5, rng=np.random.RandomState(seed)
        )
        for offset, length in offset_lengths:
            chunk_here = sorted(i - offset for i in indices if offset <= i < offset + length)
            chunk_size = max(1, int(0.15 * length))
            min_start = min(int(0.5 * length), max(0, length - chunk_size))
            assert chunk_here[0] >= min_start, "chunk start fell before start_min_fraction"

    # Default (start_min_fraction=0.0) must still be able to draw a start near 0
    # (sanity check that the new parameter didn't change the old default behavior).
    saw_near_zero_start = False
    for seed in range(50):
        indices = combined.phase_chunk_indices(chunk_fraction=0.15, rng=np.random.RandomState(seed))
        offset, length = offset_lengths[0]
        chunk_here = sorted(i - offset for i in indices if offset <= i < offset + length)
        if chunk_here[0] < int(0.05 * length):
            saw_near_zero_start = True
            break
    assert saw_near_zero_start
    print("[DEBUG] test_phase_chunk_indices_start_min_fraction_bounds_random_draw passed successfully.")


def test_assert_consistent_grid_rejects_mismatched_potentials():
    print("\n[DEBUG] Starting test_assert_consistent_grid_rejects_mismatched_potentials...")
    ref = SimpleNamespace(domain_resolution=320, box_length=9.96875, dr=0.03115)
    matching = SimpleNamespace(domain_resolution=320, box_length=9.96875, dr=0.03115)
    mismatched = SimpleNamespace(domain_resolution=100, box_length=9.96875, dr=0.03115)

    # Same grid: no error
    MultiPotentialMDDataset._assert_consistent_grid(["ref", "match"], [ref, matching])

    # Different resolution: must raise
    with pytest.raises(ValueError):
        MultiPotentialMDDataset._assert_consistent_grid(["ref", "mismatch"], [ref, mismatched])
    print("[DEBUG] test_assert_consistent_grid_rejects_mismatched_potentials passed successfully.")
