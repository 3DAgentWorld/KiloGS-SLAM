# Installation notes

Follow the [README installation steps](../README.md#installation) first. This page covers optional setup and common build issues.

## CUDA and OpenCV

The extensions need the CUDA compiler (`nvcc`), not just the PyTorch CUDA runtime. If compilation cannot find CUDA, check that `CUDA_HOME` points to your toolkit and that `nvcc --version` works.

For OpenCV installed outside the system directories, set its CMake path before running the installer:

```bash
# Directory containing OpenCVConfig.cmake; replace with your installation path.
export OpenCV_DIR=/path/to/opencv/lib/cmake/opencv4
```

The `opencv-python` package does not replace the OpenCV C++ development files (`libopencv-dev`). DBoW2 installs inside the active Conda environment.

## Build and verify

The installer compiles the bundled FastGS renderer, simple-knn, DBoW2/DPRetrieval, and sim3solve. Keep the bundled renderer: other rasterizer versions have incompatible interfaces.

These sources are included as ordinary directories, so the main repository does not need a recursive clone. The renderer includes GLM; sim3solve uses the bundled Eigen headers; DPRetrieval includes pybind11. MASt3R is downloaded separately with its recursive submodules, as shown in the README.

To reduce compiler memory usage:

```bash
MAX_JOBS=2 bash scripts/install_kilogs.sh
```

After installation, check the CUDA extensions on an idle GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/check_cuda_extensions.py
```

Dependency versions are recorded in `requirements-kilogs.txt` and `constraints-kilogs.txt`. Keep the PyTorch, xFormers and torch-scatter builds compatible when changing the environment.

## First run and offline use

Feature matching and rendering evaluation download additional weights on first use. For offline runs, populate the Torch cache beforehand; its default location is `~/.cache/torch/hub/checkpoints`, or `$TORCH_HOME/hub/checkpoints` when configured.

MASt3R works with the PyTorch RoPE fallback; compiling the optional CroCo extension is not required. Depth Anything 3 needs its own package only for scenes that select it. Model downloads and configuration paths are listed in the [README](../README.md#checkpoints).
