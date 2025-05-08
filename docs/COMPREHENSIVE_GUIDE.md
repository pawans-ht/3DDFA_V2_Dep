# 3DDFA_V2 Comprehensive Guide

## 1. Introduction

3DDFA_V2 (Towards Fast, Accurate and Stable 3D Dense Face Alignment) is an advanced project that significantly extends the capabilities of the original [3DDFA](https://github.com/cleardusk/3DDFA). It focuses on providing a robust solution for 3D dense face alignment, offering improvements in speed, accuracy, and stability. The project is packaged as `threeddfa`.

**Key Features:**

*   **Fast and Accurate Alignment:** Delivers high-quality 3D dense face alignment.
*   **Versatile Output Types:** Supports a wide range of outputs, including:
    *   2D sparse facial landmarks
    *   2D dense facial landmarks
    *   3D face mesh rendering
    *   Depth map generation
    *   Projected Normalized Coordinate Code (PNCC) map
    *   UV texture map
    *   Head pose estimation (angles)
*   **3D Model Export:** Allows exporting the 3D face mesh to `.ply` and `.obj` file formats.
*   **ONNX Runtime Support:** Integrates [ONNX Runtime](https://onnxruntime.ai/) for substantial inference speedup, especially on CPU.
*   **Real-time Capabilities:** Enables processing of video streams from files or webcams in real-time.
*   **Face Detection:** Incorporates [FaceBoxes](https://github.com/zisianw/FaceBoxes.PyTorch) for efficient face detection.

For in-depth details, please refer to the research paper and supplementary materials:
*   **Paper:** [Towards Fast, Accurate and Stable 3D Dense Face Alignment](https://guojianzhu.com/assets/pdfs/3162.pdf) (ECCV 2020)
*   **Supplementary Material:** [Link](https://guojianzhu.com/assets/pdfs/3162-supp.pdf)
*   **Main GitHub Repository:** [cleardusk/3DDFA_V2](https://github.com/cleardusk/3DDFA_V2) (This is the original repository. Please adapt if you are working from a fork).

A web-based demonstration is also available via Gradio, showcasing the project's capabilities interactively. You can try it online here: [Gradio Hub 3DDFA_V2](https://gradio.app/hub/AK391/3DDFA_V2).

## 2. Getting Started

This section will guide you through setting up the `threeddfa` package and running the demonstrations.

### 2.1. Prerequisites

*   **Operating Systems:**
    *   Linux (tested)
    *   macOS (tested)
    *   Windows (May require adjustments for C++/Cython compilation. Refer to the original project's FQA or GitHub issues for community solutions).
*   **Python:** Version `>=3.12` as specified in [`pyproject.toml`](../pyproject.toml).
*   **Core Libraries:**
    *   The necessary Python dependencies are listed in [`pyproject.toml`](../pyproject.toml) and will be installed automatically. Key dependencies include PyTorch, NumPy, OpenCV-Python, ONNXRuntime, Cython, and Gradio.
*   **Build Tools:**
    *   A C/C++ compiler (like GCC) is needed for compiling Cython extensions and the custom `render.so` library during installation.
*   **ONNX Acceleration (macOS):** If you plan to use the `--onnx` flag for acceleration on macOS, you may need to install `libomp`:
    ```bash
    brew install libomp
    ```
*   **Webcam/Video Input:** For webcam access and some video operations, `imageio-ffmpeg` is required (listed as a dependency).

### 2.2. Installation

The `threeddfa` package is designed to be installed using pip, which will also handle the compilation of necessary C and Cython extensions.

1.  **Clone the Repository:**
    You need the repository not only for installation but also to access example scripts, configuration files, example data, and pre-trained weights, as these are not distributed as part of the installable package.
    ```bash
    git clone https://github.com/cleardusk/3DDFA_V2.git # Or your fork's URL
    cd 3DDFA_V2
    ```

2.  **Install the Package:**
    It's highly recommended to use a Python virtual environment. From the root of the cloned repository (e.g., `3DDFA_V2/`), run:
    ```bash
    pip install .
    ```
    This command will:
    *   Read [`pyproject.toml`](../pyproject.toml) and [`setup.py`](../setup.py).
    *   Install all Python dependencies.
    *   Compile the Cython extensions for FaceBoxes NMS and Sim3DR.
    *   Compile `threeddfa/utils/asset/render.c` into `render.so` and include it.

    After successful installation, the `threeddfa` package and its command-line scripts (if any are defined as entry points in `setup.py`) will be available in your Python environment.

### 2.3. Quick Start: Running a Demo

Once the package is installed, you can run the demonstration scripts. **Important:** The demo scripts rely on configuration files (`configs/`), example images/videos (`examples/`), and pre-trained model weights (`weights/`) which are located in the cloned repository and are **not** installed into your Python environment's site-packages directory.

**Therefore, you should run the demo scripts from the root directory of your cloned repository.**

**Example: Processing a Still Image**

Navigate to the root of your cloned `3DDFA_V2` repository in your terminal. Then, run:

```bash
python threeddfa/demo.py -f examples/inputs/emma.jpg -o 3d --onnx
```

This command will:
*   Execute the [`threeddfa/demo.py`](../threeddfa/demo.py) script.
*   Load the input image `examples/inputs/emma.jpg`. The path is relative to your current directory (the project root).
*   Specify the output type as `3d` (3D mesh rendering).
*   Use the `--onnx` flag for ONNX-accelerated inference.

The script will process the image and, by default, display the result and save it to a path like `examples/results/emma_3d.jpg`.

**Understanding Paths in Demos:**
The demo scripts (e.g., [`threeddfa/demo.py`](../threeddfa/demo.py), [`threeddfa/demo_video.py`](../threeddfa/demo_video.py)) use relative paths for:
*   **Configuration files:** e.g., the default `-c configs/mb1_120x120.yml` in [`threeddfa/demo.py`](../threeddfa/demo.py:139) is relative to the script's location within the `threeddfa` directory.
*   **Input files:** e.g., `-f examples/inputs/trump_hillary.jpg`.
*   **Output files:** e.g., `examples/results/...`.

When you run `python threeddfa/demo.py ...` from the project root:
*   Python executes `threeddfa/demo.py`.
*   The script correctly resolves `configs/mb1_120x120.yml` because it's relative to its own location.
*   Paths like `examples/inputs/...` are resolved relative to the Current Working Directory (which is the project root).

This setup ensures that the demos work as intended when run from the project root after installing the package.

## 3. Core Usage and Demonstrations

This section details how to use the various demonstration scripts provided with 3DDFA_V2. Remember to run these commands from the root directory of your cloned repository.

### 3.1. Processing Still Images

The [`threeddfa/demo.py`](../threeddfa/demo.py) script is used for processing single still images.

**Command-Line Usage:**

```bash
python threeddfa/demo.py -f <path_to_image> -o <output_type> [options]
```

**Key Arguments:**

*   `-f, --img_fp`: Path to the input image file (e.g., `examples/inputs/emma.jpg`).
*   `-o, --opt`: Specifies the type of output to generate.
*   `--onnx`: (Flag) Use ONNXRuntime for faster inference. Highly recommended.
*   `-c, --config`: Path to the model configuration file. Defaults to `configs/mb1_120x120.yml` (relative to the script's location).
*   `-m, --mode`: `cpu` or `gpu` mode (GPU mode might require PyTorch with CUDA). Default is `cpu`.
*   `--show_flag`: (`true` or `false`) Whether to display the visualization result. Default is `true`.

**Output Types (`-o, --opt`):**

*   `2d_sparse`: Renders sparse (68 points) facial landmarks on the image.
    *   Example: `python threeddfa/demo.py -f examples/inputs/trump_hillary.jpg -o 2d_sparse --onnx`
    *   Output: `examples/results/trump_hillary_2d_sparse.jpg`
*   `2d_dense`: Renders dense facial landmarks on the image.
    *   Example: `python threeddfa/demo.py -f examples/inputs/trump_hillary.jpg -o 2d_dense --onnx`
    *   Output: `examples/results/trump_hillary_2d_dense.jpg`
*   `3d`: Renders the 3D face mesh overlaid on the image.
    *   Example: `python threeddfa/demo.py -f examples/inputs/trump_hillary.jpg -o 3d --onnx`
    *   Output: `examples/results/trump_hillary_3d.jpg`
*   `depth`: Generates and saves a depth map of the face.
    *   Example: `python threeddfa/demo.py -f examples/inputs/trump_hillary.jpg -o depth --onnx`
    *   Output: `examples/results/trump_hillary_depth.jpg`
*   `pncc`: Generates and saves a Projected Normalized Coordinate Code (PNCC) map.
    *   Example: `python threeddfa/demo.py -f examples/inputs/trump_hillary.jpg -o pncc --onnx`
    *   Output: `examples/results/trump_hillary_pncc.jpg`
*   `uv_tex`: Generates and saves a UV texture map.
    *   Example: `python threeddfa/demo.py -f examples/inputs/trump_hillary.jpg -o uv_tex --onnx`
    *   Output: `examples/results/trump_hillary_uv_tex.jpg`
*   `pose`: Estimates and visualizes the head pose (Euler angles) on the image.
    *   Example: `python threeddfa/demo.py -f examples/inputs/trump_hillary.jpg -o pose --onnx`
    *   Output: `examples/results/trump_hillary_pose.jpg`
*   `ply`: Exports the 3D face mesh to a `.ply` file.
    *   Example: `python threeddfa/demo.py -f examples/inputs/trump_hillary.jpg -o ply --onnx`
    *   Output: `examples/results/trump_hillary.ply`
*   `obj`: Exports the 3D face mesh and texture to `.obj` and `.mtl` files.
    *   Example: `python threeddfa/demo.py -f examples/inputs/trump_hillary.jpg -o obj --onnx`
    *   Output: `examples/results/trump_hillary.obj` and `examples/results/trump_hillary.mtl`

The output files are typically saved in the `examples/results/` directory.

### 3.2. Processing Videos

3DDFA_V2 provides scripts for processing video files, with options for basic frame-by-frame processing and temporal smoothing.

#### 3.2.1. Basic Video Processing

The [`threeddfa/demo_video.py`](../threeddfa/demo_video.py) script processes videos frame by frame.

**Command-Line Usage:**

```bash
python threeddfa/demo_video.py -f <path_to_video> -o <output_type> [options]
```

**Key Arguments:**

*   `-f, --video_fp`: Path to the input video file (e.g., `examples/inputs/videos/214.avi`).
*   `-o, --opt`: Output type. For this script, choices are `2d_sparse` or `3d`. Default is `2d_sparse`.
*   `--onnx`: (Flag) Use ONNXRuntime.
*   `-c, --config`: Model configuration file.
*   `-m, --mode`: `cpu` or `gpu`.

**Functionality:**
The script detects faces in the first frame and then uses a landmark-based tracking approach for subsequent frames. If tracking is lost (based on a simple bounding box size heuristic), it attempts to re-detect.

**Example:**

```bash
python threeddfa/demo_video.py -f examples/inputs/videos/214.avi -o 3d --onnx
```
This will process `214.avi`, render the 3D mesh on each frame, and save the output to `examples/results/videos/214_3d.mp4`.

#### 3.2.2. Smoothed Video Processing

For more temporally stable results, use [`threeddfa/demo_video_smooth.py`](../threeddfa/demo_video_smooth.py). This script applies a smoothing filter over a window of frames.

**Command-Line Usage:**

```bash
python threeddfa/demo_video_smooth.py -f <path_to_video> -o <output_type> [options]
```

**Key Arguments (in addition to basic video args):**

*   `-o, --opt`: Output type. Choices: `2d_sparse`, `2d_dense`, `3d`. Default: `2d_sparse`.
*   `-n_pre`: Number of previous frames to include in the smoothing window. Default: `1`.
*   `-n_next`: Number of next frames to include in the smoothing window. Default: `1`.
*   `-s, --start`: Start frame number for processing. Default: `-1` (start from beginning).
*   `-e, --end`: End frame number for processing. Default: `-1` (process until end).

**Example:**

```bash
python threeddfa/demo_video_smooth.py -f examples/inputs/videos/214.avi -o 3d --n_pre 2 --n_next 2 --onnx
```
This command processes the video with 3D mesh output, using a smoothing window of 5 frames (2 previous, current, 2 next). The output will be saved to a file like `examples/results/videos/214_3d_smooth.mp4`.

### 3.3. Real-time Webcam Application

The [`threeddfa/demo_webcam_smooth.py`](../threeddfa/demo_webcam_smooth.py) script allows real-time face alignment using your webcam, incorporating temporal smoothing.

**Command-Line Usage:**

```bash
python threeddfa/demo_webcam_smooth.py -o <output_type> [options]
```

**Key Arguments:**

*   `-o, --opt`: Output type. Choices: `2d_sparse`, `2d_dense`, `3d`. Default: `2d_sparse`.
*   `--onnx`: (Flag) Use ONNXRuntime.
*   `-c, --config`: Model configuration file.
*   `-m, --mode`: `cpu` or `gpu`.
*   `-n_pre`: Number of previous frames for smoothing. Default: `1`.
*   `-n_next`: Number of next frames for smoothing. Default: `1`.

**Functionality:**
The script accesses the default webcam (usually `<video0>`). It performs detection, tracking, and smoothing similar to the smoothed video demo, displaying the results in an OpenCV window. Press 'q' to quit.

**Example:**

```bash
python threeddfa/demo_webcam_smooth.py -o 3d --onnx
```
This will start the webcam demo, rendering a 3D face mesh in real-time.

**Note:** Ensure `imageio-ffmpeg` is installed for webcam access.

### 3.4. Gradio Web Interface

3DDFA_V2 includes a Gradio-based web interface for an interactive demonstration.

**Running the Gradio Demo:**

Execute the [`threeddfa/gradiodemo.py`](../threeddfa/gradiodemo.py) script from the project root:

```bash
python threeddfa/gradiodemo.py
```

**Functionality:**
This command will start a local web server. Open the URL provided in your terminal (usually `http://127.0.0.1:7860` or similar) in a web browser.
The interface allows you to:
*   Upload your own images.
*   Use provided example images.
*   View the 3D face mesh rendered on the input image.

The Gradio demo is hardcoded to use ONNX for performance.

## 4. Advanced Topics

This section delves into more advanced aspects of 3DDFA_V2, including model configuration, performance tuning, and an overview of its core components.

### 4.1. Model Configuration

The behavior and performance of the 3D face alignment model are controlled by YAML configuration files. These files specify parameters suchas the model architecture, input size, and pre-trained weights to use.

**Role of Configuration Files:**
All demo scripts accept a `-c` or `--config` argument to specify a configuration file. If not provided, a default configuration is used (typically `configs/mb1_120x120.yml`, relative to the script's location within the `threeddfa` package directory).

**Example Configuration File:**
The primary configuration file is [`threeddfa/configs/mb1_120x120.yml`](../threeddfa/configs/mb1_120x120.yml). It defines settings for the MobileNetV1 backbone with a 120x120 input size.

Another provided configuration is [`threeddfa/configs/mb05_120x120.yml`](../threeddfa/configs/mb05_120x120.yml), which uses a MobileNetV1 with a 0.5 width factor, resulting in a smaller and faster model.

**Specifying a Configuration:**
To use a different configuration, pass its path to the demo script:
```bash
python threeddfa/demo.py -f examples/inputs/emma.jpg -o 3d --config threeddfa/configs/mb05_120x120.yml --onnx
```
**Note:** When running demo scripts from the project root, the path to the config file should be relative to the project root if the script itself expects it that way, or relative to the script's location if the script resolves it internally (as is the case with the default config path in the demos). The demo scripts are written such that their default config paths like `configs/mb1_120x120.yml` are relative to their own location within the `threeddfa` package structure. If you provide a custom path via `-c`, ensure it's correctly specified (e.g., `threeddfa/configs/your_config.yml` if running from project root).

**Pre-trained Weights:**
The configuration files also point to the pre-trained model weights (e.g., `weights/mb1_120x120.pth`). These weight files are located in the `weights/` directory of the cloned repository.

### 4.2. Performance Considerations

Achieving optimal performance, especially for real-time applications, is crucial. 3DDFA_V2 offers several ways to enhance speed.

**ONNX Runtime:**
Using ONNX (Open Neural Network Exchange) Runtime can significantly reduce inference latency, particularly on CPUs.
*   **Enabling ONNX:** Pass the `--onnx` flag to any of the demo scripts. This will switch to using ONNX versions of the face detector (`FaceBoxes_ONNX`) and the 3DMM regressor (`TDDFA_ONNX`).
*   **Benefits:** The original [`readme.md`](../readme.md) highlights substantial speedups. For instance, with the MobileNetV1 backbone, inference time for 3DMM parameters can drop to around 1.35ms/image on a CPU with 4 threads.
*   **Dependencies:** Ensure `onnxruntime` is installed (it's a project dependency). For macOS, `libomp` might be needed for optimal multi-threading with ONNX.

**CPU Threads (`OMP_NUM_THREADS`):**
When using ONNX on CPU, the number of threads used by OpenMP can impact performance.
*   **Setting Threads:** The demo scripts, when `--onnx` is active, often set `os.environ['OMP_NUM_THREADS'] = '4'` internally.
*   **Manual Control:** You can also set this environment variable externally before running a script:
    ```bash
    export OMP_NUM_THREADS=4 # For Linux/macOS
    # set OMP_NUM_THREADS=4 # For Windows Command Prompt
    python threeddfa/demo.py ... --onnx
    ```
    Experiment with different thread counts (e.g., 1, 2, 4, or matching your CPU core count) to find the best performance on your specific hardware.

**Latency Benchmarks:**
The project includes a script to evaluate latency: [`threeddfa/latency.py`](../threeddfa/latency.py).
*   **Running the Benchmark:**
    ```bash
    python threeddfa/latency.py --onnx
    ```
*   This script measures the time taken for different parts of the pipeline (face detection, 3DMM regression, reconstruction).
*   Refer to the output of this script and the "Latency" section in the main [`readme.md`](../readme.md) for detailed performance figures on example hardware.

**GPU Mode (`-m gpu`):**
While ONNX provides significant CPU speedups, the original PyTorch models can also run on a GPU if available and PyTorch is installed with CUDA support.
*   **Enabling GPU Mode:** Pass `-m gpu` to the demo scripts. Note that this option is typically used when *not* using the `--onnx` flag, as the ONNX demos are primarily optimized for CPU.
    ```bash
    python threeddfa/demo.py -f examples/inputs/emma.jpg -o 3d -m gpu
    ```

### 4.3. Key Architectural Components

Understanding the main components of 3DDFA_V2 can be helpful for advanced usage or modification:

*   **`FaceBoxes` / `FaceBoxes_ONNX`:**
    *   Located in [`threeddfa/FaceBoxes/`](../threeddfa/FaceBoxes/).
    *   This module is responsible for detecting faces in the input image. It's based on the [FaceBoxes.PyTorch](https://github.com/zisianw/FaceBoxes.PyTorch) implementation.
    *   `FaceBoxes_ONNX.py` provides the ONNX-accelerated version.

*   **`TDDFA` / `TDDFA_ONNX`:**
    *   Core modules: [`threeddfa/TDDFA.py`](../threeddfa/TDDFA.py) and [`threeddfa/TDDFA_ONNX.py`](../threeddfa/TDDFA_ONNX.py).
    *   These classes handle the regression of 3D Morphable Model (3DMM) parameters from the detected face regions.
    *   They take the image and face bounding boxes as input and output a list of parameters (shape, expression, pose) for each face.
    *   They also include methods for reconstructing 3D vertices (`recon_vers`) from these parameters.

*   **`Sim3DR` (Similarity Transform 3D):**
    *   Located in [`threeddfa/Sim3DR/`](../threeddfa/Sim3DR/).
    *   This component, particularly its Cython part (`Sim3DR_Cython.pyx`), is involved in 3D transformations and includes a basic rasterizer for rendering the 3D mesh.

*   **Rendering Utilities:**
    *   [`threeddfa/utils/render.py`](../threeddfa/utils/render.py): Contains Python-based rendering functions.
    *   [`threeddfa/utils/render_ctypes.py`](../threeddfa/utils/render_ctypes.py): Provides an interface to a faster C-based rendering function (`render.so` compiled from `threeddfa/utils/asset/render.c`). The demos often choose one of these for visualization. The C-based version is generally preferred for speed.

*   **Other Utilities (`threeddfa/utils/`):**
    *   This directory contains various helper functions for tasks like:
        *   Depth map calculation ([`depth.py`](../threeddfa/utils/depth.py))
        *   PNCC generation ([`pncc.py`](../threeddfa/utils/pncc.py))
        *   UV texture mapping ([`uv.py`](../threeddfa/utils/uv.py))
        *   Pose visualization ([`pose.py`](../threeddfa/utils/pose.py))
        *   Serialization to .ply and .obj formats ([`serialization.py`](../threeddfa/utils/serialization.py))
        *   Landmark drawing ([`functions.py`](../threeddfa/utils/functions.py))

## 5. Understanding Outputs

3DDFA_V2 produces a variety of outputs, from 2D visualizations on images to 3D model files. This section helps in interpreting them. The example images referenced below can be found in the `docs/images/` directory of the cloned repository.

### 5.1. Visual Outputs (Images/Video Frames)

These outputs are typically generated when processing images or videos and are either displayed directly or saved as image/video files.

*   **2D Sparse Landmarks (`-o 2d_sparse`)**
    *   Description: These are a standard set of 68 facial keypoints (e.g., outlining eyes, eyebrows, nose, mouth, jawline) overlaid on the face.
    *   Interpretation: Useful for tasks like emotion recognition, basic face tracking, or as input to other facial analysis algorithms.
    *   Example: `python threeddfa/demo.py -f examples/inputs/trump_hillary.jpg -o 2d_sparse`
    *   Visual: ![2D Sparse Landmarks](images/trump_hillary_2d_sparse.jpg)

*   **2D Dense Landmarks (`-o 2d_dense`)**
    *   Description: A much larger set of landmarks providing a denser representation of the facial surface.
    *   Interpretation: Offers a more detailed 2D outline of the face shape.
    *   Example: `python threeddfa/demo.py -f examples/inputs/trump_hillary.jpg -o 2d_dense`
    *   Visual: ![2D Dense Landmarks](images/trump_hillary_2d_dense.jpg)

*   **3D Mesh Overlay (`-o 3d`)**
    *   Description: A 3D wireframe or solid mesh representing the reconstructed face shape, overlaid on the original image, aligned with the person's face.
    *   Interpretation: Provides a direct visualization of the 3D face reconstruction in the context of the 2D image.
    *   Example: `python threeddfa/demo.py -f examples/inputs/trump_hillary.jpg -o 3d`
    *   Visuals:
        *   ![3D Mesh on Emma](images/emma_3d.jpg)
        *   ![3D Mesh on Trump/Hillary](images/trump_hillary_3d.jpg)

*   **Depth Map (`-o depth`)**
    *   Description: An image where pixel intensity corresponds to the estimated depth (distance from the camera) of each point on the facial surface. Brighter pixels are typically closer, and darker pixels are further away (this can vary based on visualization).
    *   Interpretation: Represents the 3D geometry of the face from the camera's viewpoint. Useful for understanding facial structure in 3D.
    *   Example: `python threeddfa/demo.py -f examples/inputs/trump_hillary.jpg -o depth`
    *   Visual: ![Depth Map](images/trump_hillary_depth.jpg)

*   **PNCC (Projected Normalized Coordinate Code) (`-o pncc`)**
    *   Description: PNCC maps encode 3D surface information in a 2D image format. Each color channel (R, G, B) at a pixel typically represents the X, Y, Z coordinates of the corresponding 3D point in a normalized space.
    *   Interpretation: A compact representation of the 3D face shape, often used as input for other deep learning models or for tasks like face recognition robust to pose variations.
    *   Example: `python threeddfa/demo.py -f examples/inputs/trump_hillary.jpg -o pncc`
    *   Visual: ![PNCC Map](images/trump_hillary_pncc.jpg)

*   **UV Texture Map (`-o uv_tex`)**
    *   Description: A 2D image representing the "unwrapped" texture of the 3D face model. It maps the 3D surface coordinates to a 2D plane (the UV space).
    *   Interpretation: This is the texture that would be applied to the 3D mesh to give it a realistic appearance. Useful for texture analysis, editing, or re-texturing applications.
    *   Example: `python threeddfa/demo.py -f examples/inputs/trump_hillary.jpg -o uv_tex`
    *   Visual: ![UV Texture Map](images/trump_hillary_uv_tex.jpg)

*   **Head Pose Visualization (`-o pose`)**
    *   Description: Displays the estimated head pose (typically yaw, pitch, roll angles) overlaid on the image, often with axes indicating the orientation.
    *   Interpretation: Quantifies the head's orientation in 3D space.
    *   Example: `python threeddfa/demo.py -f examples/inputs/trump_hillary.jpg -o pose`
    *   Visual: ![Head Pose](images/trump_hillary_pose.jpg)

### 5.2. 3D Model Files

3DDFA_V2 can export the reconstructed 3D face model into standard file formats, allowing you to use them in 3D modeling software, game engines, or other 3D applications.

*   **PLY Files (`.ply`) (`-o ply`)**
    *   Description: The Polygon File Format (or Stanford Triangle Format) is a common format for storing 3D data from scanned objects.
    *   Content: PLY files generated by 3DDFA_V2 typically store:
        *   **Vertices:** A list of X, Y, Z coordinates for each point in the 3D mesh.
        *   **Faces (Triangles):** Definitions of how vertices are connected to form triangular faces, defining the mesh surface.
        *   Optionally, vertex colors or other properties might be included.
    *   Usage: Can be imported into most 3D viewers and modeling software (e.g., MeshLab, Blender, CloudCompare).
    *   Example: `python threeddfa/demo.py -f examples/inputs/trump_hillary.jpg -o ply`
    *   Visual (representation of a .ply file opened in a viewer): ![PLY File Example](images/ply.jpg)

*   **OBJ Files (`.obj`, `.mtl`) (`-o obj`)**
    *   Description: The Wavefront OBJ format is a widely supported 3D geometry definition file format.
    *   Content:
        *   **`.obj` file:** Contains the geometry information:
            *   `v`: Vertex coordinates (X, Y, Z).
            *   `vt`: Texture coordinates (U, V) for mapping a 2D texture onto the 3D surface.
            *   `vn`: Vertex normals (optional, for lighting calculations).
            *   `f`: Face definitions, specifying which vertices (and optionally, texture coordinates and normals) form each polygonal face (typically triangles).
            *   It may also reference a material library file (`.mtl`).
        *   **`.mtl` file (Material Template Library):** Defines material properties, such as color, specularity, and importantly, references to texture image files (e.g., a `.png` or `.jpg` file for the face texture). When 3DDFA_V2 generates an OBJ, it often extracts a texture from the input image and saves it alongside, referenced by the `.mtl` file.
    *   Usage: Broadly supported by 3D modeling software (Blender, Maya, 3ds Max), game engines (Unity, Unreal Engine), and 3D viewers. The combination of `.obj` (geometry) and `.mtl` + texture image (appearance) allows for a fully textured 3D model.
    *   Example: `python threeddfa/demo.py -f examples/inputs/trump_hillary.jpg -o obj`
    *   Visual (representation of a textured .obj file): ![OBJ File Example](images/obj.jpg)

## 6. Troubleshooting & FAQ

This section addresses common issues and frequently asked questions.

**Q1: I'm having trouble building the Cython modules (NMS, Sim3DR) or `render.so` during `pip install .`**

*   **Ensure Build Tools:** Make sure you have a C/C++ compiler (like GCC on Linux/macOS, or MSVC build tools on Windows) and Python development headers installed.
    *   Linux (Debian/Ubuntu): `sudo apt-get install build-essential python3-dev`
    *   macOS: Xcode Command Line Tools usually suffice (`xcode-select --install`).
    *   Windows: This is often the trickiest. You'll need Microsoft C++ Build Tools. Refer to Python's official documentation on compiling extensions on Windows or search for specific errors you encounter. The original 3DDFA_V2 repository's issues section might have community solutions (e.g., [this comment](https://github.com/cleardusk/3DDFA_V2/issues/12#issuecomment-697479173) for NMS on Windows).
*   **Cython Version:** Ensure your Cython version is compatible (see [`pyproject.toml`](../pyproject.toml)). `pip install .` should handle this.
*   **NumPy:** Ensure NumPy is installed and its headers are available, as it's a dependency for the Cython extensions. `pip install .` should also manage this.
*   **Check Error Logs:** Carefully examine the error messages during the `pip install .` process. They often point to missing dependencies or compiler issues.

**Q2: The demo script says "No face detected."**

*   **Image Quality:** Ensure the input image is clear, well-lit, and the face is reasonably prominent. Very small, blurry, or occluded faces might not be detected.
*   **Face Detector Thresholds:** The FaceBoxes detector has internal confidence thresholds. If faces are present but not detected, they might be below this threshold. (Advanced: Modifying detector thresholds would require code changes in `threeddfa/FaceBoxes/`).
*   **Correct File Path:** Double-check that the path provided via `-f` is correct and the image file exists and is readable.

**Q3: Tracking fails in videos or the webcam demo (e.g., mesh becomes unstable or disappears).**

*   **Fast Motion/Extreme Poses:** The tracking mechanism in the demos is relatively simple (landmark-based cropping and re-feeding to TDDFA). Very fast head movements, extreme out-of-plane rotations (e.g., head pose > 90 degrees from frontal), or significant occlusions can cause tracking to fail. The demos include a basic heuristic to re-initialize detection if the tracked bounding box becomes too small, but this isn't foolproof.
*   **Lighting Changes:** Sudden or drastic changes in lighting can affect landmark stability and thus tracking.
*   **Smoothing Parameters:** For `demo_video_smooth.py` and `demo_webcam_smooth.py`, adjusting `--n_pre` and `--n_next` might help, but excessive smoothing can cause lag.

**Q4: The landmarks around the eyes are not accurate, especially when eyes are closed.**

*   **Training Data:** As mentioned in the original `readme.md`, the model was trained primarily on the 300W-LP dataset. This dataset has limited examples of closed-eye faces. Consequently, the model's performance for closed-eye states or very expressive eye movements might be suboptimal.

**Q5: What is the training data used for the provided models?**

*   The primary training dataset is [300W-LP](https://drive.google.com/file/d/0B7OEHD3T4eCkVGs0TkhUWFN6N1k/view?usp=sharing). For more details on the training process, refer to the [original paper](https://guojianzhu.com/assets/pdfs/3162.pdf).

**Q6: How do I use a different pre-trained model or my own model?**

*   **Configuration:** You would need to:
    1.  Place your model weights file (e.g., a `.pth` file for PyTorch or `.onnx` for ONNX) in a location accessible by the scripts (e.g., the `weights/` directory).
    2.  Create or modify a YAML configuration file in `threeddfa/configs/` to point to your new model weights and specify its architecture (e.g., backbone type, input size if different).
    3.  Pass this new configuration file to the demo scripts using the `-c` option.
*   **Model Compatibility:** Ensure your custom model's architecture and output format are compatible with what the `TDDFA` class expects.

**Q7: `ImportError` or `ModuleNotFoundError` when running demo scripts.**

*   **Installation:** Ensure you have successfully installed the `threeddfa` package by running `pip install .` from the root of the cloned repository.
*   **Virtual Environment:** If you used a virtual environment for installation, make sure it's activated when you try to run the scripts.
*   **Running from Correct Directory:** Always run the demo scripts from the root directory of the cloned repository (e.g., `python threeddfa/demo.py ...`), not from within the `threeddfa` directory itself, to ensure correct relative path resolution for examples, configs, and weights.

**Q8: Issues with ONNX Runtime, especially on specific platforms.**

*   **`libomp` on macOS:** As mentioned in prerequisites, `brew install libomp` might be necessary for multi-threaded ONNX performance.
*   **ONNXRuntime Version:** Ensure you have a compatible version of `onnxruntime` (see [`pyproject.toml`](../pyproject.toml)).
*   **Platform-Specific Builds:** ONNXRuntime provides different builds (e.g., with GPU support). Ensure you have the appropriate one if you intend to use specific execution providers beyond CPU. The default setup targets CPU execution.

## 7. Citation

If you use 3DDFA_V2 in your research or work, please cite the original papers. The primary citation for 3DDFA_V2 is:

```bibtex
@inproceedings{guo2020towards,
    title =        {Towards Fast, Accurate and Stable 3D Dense Face Alignment},
    author =       {Guo, Jianzhu and Zhu, Xiangyu and Yang, Yang and Yang, Fan and Lei, Zhen and Li, Stan Z},
    booktitle =    {Proceedings of the European Conference on Computer Vision (ECCV)},
    year =         {2020}
}
```

And for the original 3DDFA work:

```bibtex
@misc{3ddfa_cleardusk,
    author =       {Guo, Jianzhu and Zhu, Xiangyu and Lei, Zhen},
    title =        {3DDFA},
    howpublished = {\url{https://github.com/cleardusk/3DDFA}},
    year =         {2018}
}
```
Please also consider starring the [original GitHub repository](https://github.com/cleardusk/3DDFA_V2) if you find this project useful.

## 8. License

This project, 3DDFA_V2, is licensed under the MIT License. You can find the full license text in the [`LICENSE`](../LICENSE) file in the root of the repository.

The MIT License is a permissive free software license, allowing for reuse within proprietary software provided all copies of the licensed software include a copy of the MIT License terms and the copyright notice.

## 9. Further Exploration

Beyond the core functionalities and demos covered in this guide, you might find the following resources and related projects of interest:

*   **Original 3DDFA_V2 Repository:** [cleardusk/3DDFA_V2](https://github.com/cleardusk/3DDFA_V2)
    *   Explore the original issues, discussions, and community contributions.
*   **FaceBoxes.PyTorch:** [zisianw/FaceBoxes.PyTorch](https://github.com/zisianw/FaceBoxes.PyTorch)
    *   The face detection module used in 3DDFA_V2 is based on this project.
*   **Dense-Head-Pose-Estimation:** [1996scarlet/Dense-Head-Pose-Estimation](https://github.com/1996scarlet/Dense-Head-Pose-Estimation)
    *   A project using TensorFlow Lite for face mesh, head pose, and landmarks, which also inspired faster mesh rendering in 3DDFA_V2.
*   **Other Related Works (mentioned in original `readme.md`):**
    *   [face3d](https://github.com/YadiraF/face3d)
    *   [PRNet](https://github.com/YadiraF/PRNet)
*   **Gradio:** [gradio.app](https://gradio.app/)
    *   The library used for creating the interactive web demo. Explore Gradio for building your own machine learning web applications.
*   **ONNX & ONNX Runtime:**
    *   [ONNX](https://onnx.ai/): Open standard for machine learning interoperability.
    *   [ONNX Runtime](https://onnxruntime.ai/): High-performance inference engine for ONNX models.

Exploring these resources can provide deeper insights into the technologies used in 3DDFA_V2 and inspire further development or applications.