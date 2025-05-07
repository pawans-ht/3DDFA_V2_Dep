# 3DDFA_V2_Dep Package Documentation

## 1. Overview
The `3ddfa-v2-dep` package provides a Python interface for 3D dense face alignment, leveraging the research and codebase of the original 3DDFA_V2 project by Jianzhu Guo et al. It offers tools for face detection, 3D Morphable Model (3DMM) parameter estimation, and utilities for rendering 3D face reconstructions. This package aims to make the powerful features of 3DDFA_V2 more accessible for integration into other Python projects.

Key features include:
*   Robust face detection.
*   Detailed 3D face mesh reconstruction from a single 2D image.
*   Estimation of 3DMM parameters (shape and expression).
*   Rendering utilities for visualizing results.

This package is based on the work presented in:
*Jianzhu Guo, Xiangyu Zhu, Yang Yang, Fan Yang, Zhen Lei, Stan Z. Li. "Towards Fast, Accurate and Stable 3D Dense Face Alignment." ECCV, 2020.*

## 2. Installation

### 2.1. Prerequisites
Before installing `3ddfa-v2-dep`, ensure you have the following dependencies installed on your system:

*   **Python:** Version 3.12 or higher.
*   **`uv`:** Recommended for a faster and more reliable installation experience. You can install `uv` by following the instructions on its [official website](https://github.com/astral-sh/uv).
*   **C/C++ Compiler:** A working C/C++ compiler is required for building some of the underlying C/Cython extensions (e.g., GCC, Clang on Linux/macOS, MSVC on Windows).
*   **Python Development Headers:** These are necessary for compiling Python C-extensions (e.g., `python3-dev` or `python3-devel` on Linux).
*   **Git:** For cloning the repository.

### 2.2. Installing the Package

1.  **Clone the Repository:**
    Open your terminal and clone the repository. If you are using the original repository, the command is:
    ```bash
    git clone https://github.com/cleardusk/3DDFA_V2.git
    ```
    If you have forked the repository or are using a specific version, replace the URL accordingly. Note that the original repository is named `3DDFA_V2`.

2.  **Navigate to the Repository Root:**
    Change your current directory to the cloned repository's root. This might be `3DDFA_V2` if you cloned the original, or `3DDFA_V2_Dep` if your local copy is named that.
    ```bash
    cd 3DDFA_V2 
    # Or if your directory is named 3DDFA_V2_Dep:
    # cd 3DDFA_V2_Dep
    ```

3.  **Install using `uv`:**
    It's recommended to install the package in a virtual environment. `uv` can create one for you.
    ```bash
    uv venv # Create a virtual environment (e.g., .venv)
    source .venv/bin/activate # Activate the virtual environment (Linux/macOS)
    # .venv\Scripts\activate # Activate (Windows PowerShell)
    
    uv pip install .
    ```

4.  **Editable Install (for Development):**
    If you plan to modify the package's source code, you can install it in editable mode:
    ```bash
    uv pip install -e .
    ```

## 3. Getting Started: Quick Example

Here's a minimal Python script to get you started with face detection and 3DMM parameter extraction using `3ddfa-v2-dep`. Ensure you have an image file accessible; this example assumes an image at `examples/inputs/emma.jpg` relative to the project root.

```python
import cv2 # OpenCV for image loading
import numpy as np
from threeddfa_v2_dep.TDDFA import TDDFA
from threeddfa_v2_dep.FaceBoxes.FaceBoxes import FaceBoxes
from threeddfa_v2_dep.utils.tddfa_util import TDDFA_ONNX # For default config loading

# 0. Initialize TDDFA_ONNX to load default configurations
# This step might be necessary to correctly locate model and config files
# if they are bundled with the package and not specified directly.
cfg = TDDFA_ONNX.get_default_config()

# 1. Initialize FaceBoxes for face detection
# Ensure the model weights path is correct or handled by the package.
# For the packaged version, FaceBoxes might load its model internally.
fb = FaceBoxes() 

# 2. Initialize TDDFA for 3D face alignment
# Pass the loaded configuration.
# Ensure model paths in the config are correct or handled by the package.
tddfa = TDDFA(gpu_mode=False, **cfg)

# 3. Load a sample image
# Make sure 'examples/inputs/emma.jpg' exists or provide your own image path.
img_path = 'examples/inputs/emma.jpg' 
try:
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {img_path}")
except Exception as e:
    print(f"Error loading image: {e}")
    print("Please ensure you have an image at 'examples/inputs/emma.jpg' or provide a valid path.")
    exit()

# 4. Detect faces in the image
face_boxes = fb(img)
if not face_boxes:
    print("No faces detected.")
    exit()

print(f"Detected {len(face_boxes)} face(s).")

# 5. Get 3DMM parameters and reconstructed vertices for the detected faces
# The tddfa object is callable. It takes the original image and detected bounding boxes.
param_lst, roi_box_lst = tddfa(img, face_boxes)

if not param_lst:
    print("Could not extract 3DMM parameters.")
    exit()

# 6. Reconstruct vertices (3D mesh)
# dense_flag=True for dense mesh, False for sparse
# reader_flag=True to enable 68 landmarks reconstruction
ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=True, reader_flag=True) 

print(f"Extracted 3DMM parameters for {len(param_lst)} face(s).")
# Example: Print the shape of the vertices for the first detected face
if ver_lst:
    print(f"Shape of reconstructed vertices for the first face: {ver_lst[0].shape}")
    # param_lst contains [alpha, beta, R, t2d, s], which are shape, expression, rotation, translation, and scale parameters.
    # print(f"Parameters for the first face (shape, expr, R, t2d, s): {param_lst[0]}")
else:
    print("Could not reconstruct vertices.")

```
**Note:** You might need to adjust paths to models or configuration files if they are not automatically found by the package. The `TDDFA_ONNX.get_default_config()` helper is intended to simplify this by providing default paths that should work with a standard package installation.

## 4. Core Components & API

### 4.1. Face Detection (`FaceBoxes`)

*   **Module:** `threeddfa_v2_dep.FaceBoxes.FaceBoxes`
*   **Class:** `FaceBoxes`
    *   **`__init__(self, timer_flag=False)`**
        *   Initializes the FaceBoxes detector.
        *   `timer_flag`: If `True`, prints processing time.
        *   The model weights are typically loaded internally.
    *   **`__call__(self, img_)` -> `dets`**
        *   Performs face detection on the input image.
        *   `img_`: Input image (NumPy array, BGR format).
        *   Returns `dets`: A list of detected bounding boxes. Each box is `[x_min, y_min, x_max, y_max, confidence_score]`.

*   **Example:**
    ```python
    from threeddfa_v2_dep.FaceBoxes.FaceBoxes import FaceBoxes
    import cv2 # Using OpenCV for image loading

    # Initialize detector
    fb = FaceBoxes()

    # Load an image
    img = cv2.imread("path/to/your/image.jpg")
    if img is None:
        print("Error: Image not found or could not be loaded.")
    else:
        # Detect faces
        face_boxes = fb(img)

        if face_boxes:
            print(f"Found {len(face_boxes)} faces:")
            for i, box in enumerate(face_boxes):
                print(f"  Face {i+1}: Box={box[:4]}, Confidence={box[4]:.2f}")
        else:
            print("No faces detected.")
    ```

### 4.2. 3D Dense Face Alignment (`TDDFA`)

*   **Module:** `threeddfa_v2_dep.TDDFA`
*   **Class:** `TDDFA`
    *   **`__init__(self, gpu_mode=False, **kvs)`**
        *   Initializes the TDDFA model for 3D face alignment.
        *   `gpu_mode`: If `True`, attempts to use GPU for ONNX models. Ensure ONNXRuntime-GPU is installed and CUDA is configured.
        *   `**kvs`: Keyword arguments, typically loaded from a YAML configuration file. Important keys include:
            *   `model_fp`: Path to the ONNX model file (e.g., `mb1_120x120.onnx`).
            *   `config_fp`: Path to the YAML configuration file (e.g., `configs/mb1_120x120.yml`).
            *   Other parameters defined in the YAML config like `bfm_fp`, `size`, `num_params`, etc.
            *   The package attempts to provide default paths. See `threeddfa_v2_dep.utils.tddfa_util.TDDFA_ONNX.get_default_config()`.
    *   **`__call__(self, img_ori, objs, **kvs)` -> `param_lst`, `roi_box_lst`**
        *   Processes the input image and detected face bounding boxes to estimate 3DMM parameters.
        *   `img_ori`: Original input image (NumPy array, BGR).
        *   `objs`: List of face bounding boxes (e.g., from `FaceBoxes`). Can also be a list of 68-point landmarks.
        *   `**kvs`: Additional keyword arguments.
        *   Returns:
            *   `param_lst`: List of 3DMM parameters for each detected face. Each `param` is a NumPy array containing coefficients for shape, expression, rotation, 2D translation, and scale.
            *   `roi_box_lst`: List of refined bounding boxes used for cropping faces.
    *   **`recon_vers(self, param_lst, roi_box_lst, dense_flag=True, reader_flag=True, **kvs)` -> `ver_lst`**
        *   Reconstructs 3D vertices from the estimated 3DMM parameters.
        *   `param_lst`: List of 3DMM parameters from the `__call__` method.
        *   `roi_box_lst`: List of ROI boxes from the `__call__` method.
        *   `dense_flag`: If `True`, reconstructs a dense mesh. If `False`, reconstructs a sparse mesh (typically landmarks).
        *   `reader_flag`: If `True`, reconstructs 68 landmarks.
        *   `**kvs`: Additional keyword arguments.
        *   Returns `ver_lst`: List of reconstructed 3D vertices (NumPy arrays, shape `(N, 3)`) for each face.

*   **Example:**
    ```python
    from threeddfa_v2_dep.TDDFA import TDDFA
    from threeddfa_v2_dep.FaceBoxes.FaceBoxes import FaceBoxes # Assuming FaceBoxes is used for detection
    from threeddfa_v2_dep.utils.tddfa_util import TDDFA_ONNX # For default config
    import cv2

    # Assume img (loaded image) and face_boxes (from FaceBoxes) are available
    # img = cv2.imread("path/to/your/image.jpg")
    # fb = FaceBoxes()
    # face_boxes = fb(img)

    # Load default configuration
    cfg = TDDFA_ONNX.get_default_config()
    
    # Initialize TDDFA
    # Set gpu_mode=True if a compatible GPU and ONNXRuntime-GPU are available
    tddfa = TDDFA(gpu_mode=False, **cfg)

    if img is not None and face_boxes:
        # Get 3DMM parameters
        param_lst, roi_box_lst = tddfa(img, face_boxes)

        if param_lst:
            # Reconstruct dense 3D vertices
            vertices_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=True)
            if vertices_lst:
                 print(f"Reconstructed dense vertices for first face: {vertices_lst[0].shape}")

            # Reconstruct 68 landmarks (sparse vertices)
            landmarks_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=False, reader_flag=True)
            if landmarks_lst:
                print(f"Reconstructed 68 landmarks for first face: {landmarks_lst[0].shape}")
        else:
            print("Could not extract 3DMM parameters.")
    # else:
    #     print("Image not loaded or no faces detected.")
    ```

### 4.3. Rendering Utilities

The package includes utilities for rendering the reconstructed 3D faces.

#### 4.3.1. Cython-based Rasterizer (`Sim3DR`)

*   **Module:** `threeddfa_v2_dep.Sim3DR.Sim3DR`
*   This module provides a Cython-optimized rasterizer for creating 2D projections of 3D meshes.
*   **Functions:**
    *   **`get_normal(vertices, triangles)`**
        *   Calculates vertex normals for a mesh.
        *   `vertices`: NumPy array of vertex coordinates `(n_ver, 3)`.
        *   `triangles`: NumPy array of triangle indices `(n_tri, 3)`.
        *   Returns: Vertex normals `(n_ver, 3)`.
    *   **`rasterize(vertices, triangles, colors, bg=None, height=None, width=None, channel=3, reverse=False)`**
        *   Renders a 3D mesh to a 2D image.
        *   `vertices`: NumPy array of vertex coordinates `(n_ver, 3)`.
        *   `triangles`: NumPy array of triangle indices `(n_tri, 3)`.
        *   `colors`: NumPy array of vertex colors `(n_ver, 3)` or texture coordinates.
        *   `bg`: Optional background image (NumPy array).
        *   `height`, `width`: Dimensions of the output image. If `None`, inferred from `bg` or defaults.
        *   `channel`: Number of color channels (typically 3 for RGB).
        *   `reverse`: Boolean, internal flag.
        *   Returns: Rendered image as a NumPy array.

*   **Example (Conceptual):**
    Direct use of `rasterize` requires careful setup of vertices, triangles, colors, and camera parameters (which are implicitly handled when `vertices` are already projected).
    ```python
    # from threeddfa_v2_dep.Sim3DR.Sim3DR import rasterize, get_normal
    # # Assume:
    # # projected_vertices: (N, 3) array, where Z is depth. Already transformed by camera projection.
    # # triangles: (M, 3) array of triangle indices.
    # # vertex_colors: (N, 3) array of RGB colors for each vertex.
    # # background_image: Optional (H, W, 3) NumPy array.

    # if 'projected_vertices' in locals() and 'triangles' in locals() and 'vertex_colors' in locals():
    #     # Normals might be used for lighting, not directly by this rasterize function
    #     # normals = get_normal(projected_vertices, triangles) 
        
    #     # Render the mesh
    #     # rendered_img = rasterize(
    #     #     projected_vertices, 
    #     #     triangles, 
    #     #     vertex_colors, 
    #     #     bg=background_image, 
    #     #     height=background_image.shape[0] if background_image is not None else 480, 
    #     #     width=background_image.shape[1] if background_image is not None else 640
    #     # )
    #     # cv2.imshow("Sim3DR Render", rendered_img)
    #     # cv2.waitKey(0)
    #     pass # Placeholder for actual usage
    ```

#### 4.3.2. CTypes-based Renderer (`utils.render_ctypes`)

*   **Module:** `threeddfa_v2_dep.utils.render_ctypes`
*   This module provides an alternative renderer using CTypes to interface with a compiled C library. The compiled library (`render.so` or `render.dll`) should be bundled with the package.
*   **Class:** `TrianglesMeshRender`
    *   **`__init__(self, clibs_path=None, light_opt_dict=..., direction_opt_dict=..., ambient_opt_dict=...)`**
        *   Initializes the CTypes-based renderer.
        *   `clibs_path`: Path to the compiled C library. The package now handles this internally by attempting to find `render.so`/`render.dll` within the package structure. Users generally do not need to set this.
        *   `light_opt_dict`, `direction_opt_dict`, `ambient_opt_dict`: Dictionaries to configure lighting properties. Default values are provided if not specified.
    *   **`__call__(self, vertices_list, triangles, bg_img, **kwargs)`**
        *   Renders one or more meshes onto a background image.
        *   `vertices_list`: A list of vertex arrays. Each vertex array is `(N, 3)`.
        *   `triangles`: Triangle indices `(M, 3)`.
        *   `bg_img`: Background image (NumPy array).
        *   Returns: Image with rendered meshes.

*   **Function:** `render(img, ver_lst, tri, alpha=0.6, r_lst=None, t_lst=None, **kvs)`
    *   A higher-level utility function that often simplifies rendering.
    *   `img`: Background image (NumPy array).
    *   `ver_lst`: List of vertex arrays (output from `tddfa.recon_vers`).
    *   `tri`: Triangle indices. This is typically loaded from a file like `bfm_tri.mat` or `tri.mat`. See note below.
    *   `alpha`: Transparency for rendering.
    *   `r_lst`, `t_lst`: Optional lists of rotation and translation parameters if vertices are not already transformed.
    *   `**kvs`: Additional arguments like `show_flag`, `wfp` (write file path), `verbose_flag`.
    *   Returns: Image with rendered meshes.

*   **Example:**
    ```python
    from threeddfa_v2_dep.utils.render_ctypes import render as render_ctypes
    from threeddfa_v2_dep.utils.tddfa_util import load_model_bfm # Example utility for loading BFM data
    # Assume:
    # img: Original image (NumPy array)
    # vertices_lst: List of (N,3) vertex arrays from tddfa.recon_vers()

    # The 'tri' data (triangle mesh connectivity) is crucial.
    # In the original 3DDFA_V2, this is often loaded from 'tri.mat' (part of the BFM model).
    # For library users:
    # 1. This file might be bundled with the '3ddfa-v2-dep' package.
    # 2. It might need to be downloaded separately and its path provided.
    # 3. A higher-level API within the package might load it automatically.
    # Check package documentation or examples for the recommended way to obtain 'tri'.
    # As an example, if 'configs/bfm_utils/tri.mat' is accessible:
    # tri_path = 'configs/bfm_utils/tri.mat' # Adjust path as needed
    # try:
    #     tri = load_model_bfm(tri_path) # This function loads .mat files
    # except FileNotFoundError:
    #     print(f"Triangle data not found at {tri_path}. Rendering will be skipped.")
    #     tri = None

    # if img is not None and 'vertices_lst' in locals() and vertices_lst and tri is not None:
    #     rendered_image_ctypes = render_ctypes(img.copy(), vertices_lst, tri, alpha=0.6)
    #     # cv2.imshow("CTypes Render", rendered_image_ctypes)
    #     # cv2.waitKey(0)
    #     # cv2.destroyAllWindows()
    #     print("CTypes rendering example executed (display commented out).")
    # else:
    #     print("Skipping CTypes rendering example due to missing inputs.")
    ```
    **Note on `tri` (Triangle Data):** The `tri` data, representing the mesh topology (how vertices are connected to form triangles), is essential for rendering. This is typically part of the Basel Face Model (BFM). Users of the `3ddfa-v2-dep` package should verify how this data is provided or accessed. It might be bundled (e.g., accessible via a utility function within the package that loads `configs/BFM_UV.mat` or a similar file containing triangle info) or require a path to a `tri.mat` file. The `TDDFA` class initialization (via `**kvs` from config) might also load `tri` internally if `bfm_fp` points to a model that includes it.

## 5. Advanced Usage & Examples

The `3ddfa-v2-dep` package can be used for more advanced scenarios beyond single image processing. The original repository contains several demo scripts (`demo.py`, `demo_video.py`, `demo_webcam_smooth.py`) that showcase various capabilities.

To adapt these for library usage:
*   Focus on importing the necessary classes (`TDDFA`, `FaceBoxes`) and functions (`render_ctypes`) from `threeddfa_v2_dep`.
*   Initialize these classes as shown in the API examples.
*   Pass data (images, video frames) to the object instances.
*   For example, in a video processing loop, you would call `fb(frame)` and then `tddfa(frame, boxes)` for each frame.

Refer to the examples in the original repository and adapt the core logic using the packaged components.

## 6. Configuration

The `TDDFA` class relies on configuration parameters, typically specified in YAML files (e.g., `configs/mb1_120x120.yml` from the original repository). These files define:
*   Paths to ONNX models (`model_fp`).
*   Path to Basel Face Model data (`bfm_fp`).
*   Image processing parameters (e.g., `size`, `center`, `std`).
*   Number of 3DMM parameters (`num_params`).
*   And other model-specific settings.

When initializing `TDDFA`:
```python
# from threeddfa_v2_dep.TDDFA import TDDFA
# from threeddfa_v2_dep.utils.tddfa_util import _parse_param_batch # If loading params directly
# import yaml

# Option 1: Use default config helper (recommended)
# from threeddfa_v2_dep.utils.tddfa_util import TDDFA_ONNX
# cfg = TDDFA_ONNX.get_default_config() # This loads a predefined config
# tddfa = TDDFA(gpu_mode=False, **cfg)

# Option 2: Load a custom YAML config file
# config_path = "path/to/your/custom_config.yml"
# try:
#     with open(config_path, 'r') as f:
#         custom_cfg = yaml.safe_load(f)
#     # Ensure paths within custom_cfg (like model_fp, bfm_fp) are correct
#     tddfa_custom = TDDFA(gpu_mode=False, **custom_cfg)
# except Exception as e:
#     print(f"Error loading custom config: {e}")
```
The package aims to bundle default models and configurations, accessible via helpers like `TDDFA_ONNX.get_default_config()`. If you use custom models or configurations, ensure all paths within the YAML file are correct and accessible.

## 7. Troubleshooting / FAQ

*   **Build Issues:**
    *   **Missing C/C++ Compiler:** Ensure a compiler (GCC, Clang, MSVC) is installed and in your system's PATH.
    *   **Python Development Headers Not Found:** Install Python development headers (e.g., `sudo apt-get install python3-dev` on Debian/Ubuntu, `sudo yum install python3-devel` on CentOS/RHEL).
    *   **Cython Errors:** Ensure Cython is installed (`uv pip install cython`) and is compatible with your Python and compiler versions. Errors during the build of `Sim3DR_Cython.pyx` or `cpu_nms.pyx` usually point to toolchain issues.

*   **Runtime Issues:**
    *   **`render.so` / `render.dll` Loading Failed:** The CTypes renderer (`utils.render_ctypes`) depends on a compiled shared library. The `3ddfa-v2-dep` package should bundle this and locate it automatically. If you encounter errors like "OSError: render.so: cannot open shared object file", it might indicate an issue with the package build or installation.
    *   **ONNX Model Issues:**
        *   Ensure the ONNX model files (e.g., `mb1_120x120.onnx`) and face detector weights (`FaceBoxesProd.pth`) are correctly located. The package should manage this, but custom configurations might require careful path management.
        *   If using `gpu_mode=True`, ensure `onnxruntime-gpu` is installed and your CUDA/cuDNN setup is correct.
    *   **LibOMP Related Warnings/Errors (macOS):** Sometimes, OpenMP can cause issues on macOS if multiple versions are present. This might manifest as warnings or crashes. Setting environment variables like `KMP_DUPLICATE_LIB_OK=TRUE` can sometimes help as a workaround, but a proper fix involves ensuring a consistent OpenMP environment.

## 8. Project Structure Overview

The `3ddfa_v2_dep` package wraps the core functionalities of 3DDFA_V2. Key modules include:

*   **`threeddfa_v2_dep.TDDFA` (`TDDFA.py`):** Contains the main `TDDFA` class for 3D face alignment.
*   **`threeddfa_v2_dep.FaceBoxes` (`FaceBoxes/FaceBoxes.py`):** Provides the `FaceBoxes` class for face detection.
*   **`threeddfa_v2_dep.Sim3DR` (`Sim3DR/Sim3DR.py`):** Cython-based 3D rasterizer.
*   **`threeddfa_v2_dep.utils` (`utils/`):** A collection of utility functions:
    *   `render_ctypes.py`: CTypes-based rendering.
    *   `tddfa_util.py`: Helper functions for TDDFA, including ONNX model handling and configuration parsing.
    *   Other utilities for I/O, image processing, etc.
*   **`threeddfa_v2_dep.configs`:** (If bundled) Default configuration files and model parameters.
*   **`threeddfa_v2_dep.weights`:** (If bundled) Default model weights.

## 9. Citing This Work

If you use the `3ddfa-v2-dep` package or the underlying 3DDFA_V2 methodology in your research, please cite the original paper:

```
@inproceedings{guo2020towards,
  title={Towards Fast, Accurate and Stable 3D Dense Face Alignment},
  author={Guo, Jianzhu and Zhu, Xiangyu and Yang, Yang and Yang, Fan and Lei, Zhen and Li, Stan Z.},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2020}
}
```

You may also refer to the original GitHub repository: [https://github.com/cleardusk/3DDFA_V2](https://github.com/cleardusk/3DDFA_V2)

## 10. License

This package is distributed under the MIT License. Please see the `LICENSE` file in the repository for more details.