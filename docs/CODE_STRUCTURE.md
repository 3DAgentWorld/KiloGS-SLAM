# Code structure

- `slam.py` resolves CLI overrides and starts full SLAM or frontend-only execution.
- `utils/dataset.py` selects frames and prepares separate tracking and rendering resolutions.
- `utils/slam_frontend.py` coordinates image-motion gating, metric depth and pose tracking.
- `utils/visual_odometry.py` handles lightweight matching, pose estimation, scale recovery and loop verification. `utils/mast3r_matching.py` supplies hybrid correspondences; loop matching stays lightweight.
- `utils/frontend_only.py` runs pose tracking without Gaussian mapping processes.
- `utils/loop_closure/` performs asynchronous retrieval and pose-graph correction.
- `utils/slam_backend.py` manages keyframe mapping, loop updates and color refinement.
- `gaussian_splatting/scene/scaffold_model.py` manages multilevel anchors, offsets, densification, pruning and optimizer state.
- `gaussian_splatting/gaussian_renderer/` connects the model to the modified FastGS CUDA renderer.
- `scripts/run_dataset.py` runs scenes sequentially, records GPU memory, evaluates poses and writes per-run logs. The three root shell scripts supply the dataset and scene list.
- `pose_eval/` evaluates trajectories; `summary_result.py` produces the original metrics-by-scene CSV layout.

Dataset defaults live in `configs/<dataset>/base_config.yaml`; scene YAML files override calibration, data paths and scene-specific settings. CLI mode and loop switches take precedence over YAML. `Hierarchical.color_refinement_iter` controls refinement steps per stored keyframe.

Third-party source and license notices are retained in their respective directories.
