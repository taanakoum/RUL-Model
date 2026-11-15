"""Classification of legacy tests; new tests declare their level and requirements.

Keep historic file paths usable in job scripts and logs. Levels describe the work
performed, while data/weights/gpu/slow describe independent execution requirements.
"""

from __future__ import annotations

LEVELS = frozenset({"unit", "integration", "e2e"})

COMPATIBILITY_TESTS = {
    "test_manifest_and_aggregate_on_a_toy_study",
    "test_per_arm_seeds_reuse_and_parallel_seeds",
    "test_gate3_end_to_end_on_the_toy",
    "test_the_ensemble_averages_the_seeds",
}

# filename: (default level, verification requirements)
LEGACY = {
    "test_audit.py": ("unit", ("P3", "P9")),
    "test_audit_fixes.py": ("integration", ("P8", "P14", "P15", "P20", "P21", "P22")),
    "test_baselines.py": ("unit", ("P16",)),
    "test_cache.py": ("integration", ("P8",)),
    "test_causality.py": ("integration", ("P10", "P11", "P12")),
    "test_channels.py": ("unit", ("P8",)),
    "test_degradation.py": ("unit", ("P16",)),
    "test_encoders.py": ("unit", ("P8",)),
    "test_evaluation.py": ("unit", ("P17",)),
    "test_failure.py": ("unit", ("P5",)),
    "test_feature_overrides.py": ("unit", ("P9", "D24")),
    "test_folds.py": ("unit", ("P6",)),
    "test_gate3.py": ("integration", ("P14", "P15", "P20", "D29")),
    "test_images.py": ("unit", ("P1", "P8")),
    "test_inference.py": ("unit", ("P15", "D23")),
    "test_loader.py": ("integration", ("P1",)),
    "test_metrics.py": ("unit", ("P17",)),
    "test_models.py": ("unit", ("P11", "P12", "D21", "D22")),
    "test_normalization.py": ("unit", ("P7",)),
    "test_omnifuser_rul.py": ("integration", ("P10", "P14", "P16", "P18", "P21", "D20")),
    "test_padded_images.py": ("integration", ("P10", "P11", "D9")),
    "test_probe.py": ("unit", ("P9",)),
    "test_schema.py": ("unit", ("P22",)),
    "test_segments.py": ("unit", ("P4",)),
    "test_stats.py": ("unit", ("P17",)),
    "test_store.py": ("integration", ("P3", "P4", "P5", "P8")),
    "test_targets.py": ("unit", ("P5",)),
    "test_timeline.py": ("unit", ("P3",)),
    "test_tokens.py": ("integration", ("P7", "P10", "P13")),
    "test_training.py": ("integration", ("P14", "P20")),
    "test_transfer_contract.py": ("integration", ("P22",)),
    "test_trim.py": ("unit", ("P2",)),
    "test_uncertainty.py": ("unit", ("P15",)),
}

UNIT_OVERRIDES = {
    "test_learning_rate_schedule_phases",
    "test_gate_decisions",
    "test_student_t_head_ranges_and_nll",
    "test_windows_and_resampling",
    "test_decode_record_id_accepts_file_names",
    "test_duplicate_groups_within_runs_only",
    "test_inner_split_never_touches_validation_or_test",
    "test_fit_q_scores_only_the_masked_points",
    "test_pooled_and_without_thinnest_summaries_use_the_points",
    "test_select_epoch_uses_three_consecutive_values",
    "test_the_ensemble_averages_the_seeds",
    "test_shapley_values_are_exact",
    "test_two_modalities_are_judged_on_a_preset_paired_test",
    "test_the_study_arms",
    "test_the_training_loss_moves_only_the_earlier_prediction",
    "test_a_stuck_later_prediction_moves_the_earlier_one_only_a_little",
    "test_generic_code_never_imports_an_adapter",
    "test_generic_code_has_no_nonastreda_vocabulary",
}

E2E_OVERRIDES = {
    "test_fit_selects_the_smoothed_minimum_and_writes_its_outputs",
    "test_same_seed_same_weights_and_resume_after_a_kill",
    "test_manifest_and_aggregate_on_a_toy_study",
    "test_per_arm_seeds_reuse_and_parallel_seeds",
    "test_a_finished_fit_of_another_computation_is_never_skipped",
    "test_a_checkpoint_of_another_computation_is_not_resumed",
    "test_an_image_free_dataset_trains_and_predicts",
    "test_aggregation_flags_mixed_code_versions",
    "test_inner_fit_reproduces_the_outer_fit_at_its_selected_step",
    "test_gate3_end_to_end_on_the_toy",
    "test_incomplete_study_does_not_freeze",
    "test_a_fit_learns_the_toy_rul_and_a_shuffled_control_does_not",
    "test_predictions_cover_every_point_with_the_grid_flag_and_runs_resume",
    "test_a_fine_fit_predicts_every_1s_point_and_selects_on_the_grid",
    "test_the_same_seed_gives_the_same_predictions",
    "test_inside_a_fit_the_temporal_terms_move_only_earlier_predictions",
    "test_results_report_progress_scores_and_choices_before_the_freeze",
    "test_results_test_read_needs_the_freeze_and_a_complete_study",
    "test_the_run_report_keeps_the_test_sealed_before_the_freeze",
    "test_the_run_report_with_unseal_and_after_the_freeze",
    "test_the_mode_comparison_finds_a_real_difference",
    "test_toy_feature_check_and_baselines",
    "test_score_feature_baselines_scores_validation_units_only",
}


def legacy_level(filename: str, name: str, real_dependency: bool) -> str:
    if name in E2E_OVERRIDES:
        return "e2e"
    if real_dependency:
        return "integration"
    if name in UNIT_OVERRIDES:
        return "unit"
    if filename == "test_degradation.py" and name.startswith(("test_b4_", "test_r1_")):
        return "integration"
    return LEGACY[filename][0]
