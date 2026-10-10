#!/usr/bin/env bash
# Run inside the kilogs-slam environment. CUDA toolkit and libopencv-dev required.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${CONDA_PREFIX:?Activate kilogs-slam first}"
: "${CUDA_HOME:?Set CUDA_HOME to the CUDA 11.8 toolkit}"
export PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS=${MAX_JOBS:-4}
export CMAKE_BUILD_PARALLEL_LEVEL=$MAX_JOBS
# Build release extensions even when a parent shell exports DEBUG=release.
export DEBUG=0
python -m pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements-kilogs.txt -c constraints-kilogs.txt
python -m pip install xformers==0.0.24 --index-url https://download.pytorch.org/whl/cu118
python -m pip install torch-scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.2.0+cu118.html
python -m pip install --no-build-isolation ./submodules/simple-knn
# Required modified renderer: DO NOT install the similarly named legacy submodule.
python -m pip install --no-build-isolation ./submodules/diff-gaussian-rasterization_fastgs
cmake -S DBoW2 -B build/DBoW2 -DBUILD_Demo=OFF -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX" ${OpenCV_DIR:+-DOpenCV_DIR="$OpenCV_DIR"}
cmake --build build/DBoW2 --parallel "$MAX_JOBS"
cmake --install build/DBoW2
export CMAKE_PREFIX_PATH="$CONDA_PREFIX${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
export CMAKE_ARGS="-DCMAKE_PREFIX_PATH=$CONDA_PREFIX -DCMAKE_INSTALL_RPATH=$CONDA_PREFIX/lib"
if [[ -n ${OpenCV_DIR:-} ]]; then
  export CMAKE_ARGS="$CMAKE_ARGS -DOpenCV_DIR=$OpenCV_DIR"
fi
python -m pip install --no-build-isolation ./DPRetrieval
python -m pip install --no-build-isolation .
python -m pip install --no-deps ./LightGlue
python -m pip check
