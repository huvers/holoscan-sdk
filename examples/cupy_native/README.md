# CuPy Native

This minimal application multiplies two randomly generated matrices on the GPU, to showcase the use of CuPy with holoscan.

## Run instructions

* **using python wheel**:
  ```bash
  # [Prerequisite] Download example .py file below to `APP_DIR`
  # [Optional] Start the virtualenv where holoscan is installed
  python3 <APP_DIR>/matmul.py
  ```
* **from NGC container**:
  ```bash
  python3 /opt/nvidia/holoscan/examples/cupy_native/matmul.py
  ```
* **source (dev container)**:
  ```bash
  ./run launch # optional: append `install` for install tree
  python3 ./examples/cupy_native/matmul.py
  ```
* **source (local env)**:
  ```bash
  python3 -m pip install cupy-cuda12x
  export PYTHONPATH=${BUILD_OR_INSTALL_DIR}/python/lib
  python3 ${BUILD_OR_INSTALL_DIR}/examples/cupy_native/matmul.py
  ```
