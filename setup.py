import os
import subprocess
import numpy
from setuptools import setup, Extension, find_packages
from setuptools.command.build_py import build_py as _build_py

# Get the long description from the README file
with open("readme.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Define Python dependencies from pyproject.toml
# Dependencies and other metadata like name, version, python_requires
# will be sourced from pyproject.toml by setuptools.

ext_modules = [
    Extension(
        name="threeddfa.FaceBoxes.utils.nms.cpu_nms",
        sources=["threeddfa/FaceBoxes/utils/nms/cpu_nms.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-Wno-cpp", "-Wno-unused-function"], # from original build.py
    ),
    Extension(
        name="threeddfa.Sim3DR.Sim3DR_Cython",
        sources=["threeddfa/Sim3DR/lib/rasterize.pyx", "threeddfa/Sim3DR/lib/rasterize_kernel.cpp"],
        language='c++',
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-std=c++11"], # from original setup.py
    ),
    # utils.asset.render is a standard C library, not a Python extension.
    # It will be compiled separately and included as package_data.
]

# Custom command to build render.so
class BuildRenderSOCommand(_build_py):
    """Custom build command to compile render.c into render.so."""
    def run(self):
        # First, run the original build_py command
        _build_py.run(self)

        # Now, compile render.c
        compile_command = [
            "gcc",
            "-shared",
            "-Wall",
            "-O3",
            "render.c",
            "-o",
            "render.so",
            "-fPIC"
        ]
        asset_dir = os.path.join(os.path.dirname(__file__), 'threeddfa', 'utils', 'asset')
        
        # Check if render.c exists before trying to compile
        render_c_path = os.path.join(asset_dir, 'render.c')
        if not os.path.exists(render_c_path):
            print(f"Warning: {render_c_path} not found. Skipping compilation of render.so.")
            return

        print(f"Compiling {render_c_path} into {os.path.join(asset_dir, 'render.so')}")
        try:
            subprocess.check_call(compile_command, cwd=asset_dir)
            print("render.so compiled successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error compiling render.so: {e}")
            # Optionally, re-raise the error or handle it as a build failure
            # raise BuildFailed(f"Failed to compile render.so: {e}")
        except FileNotFoundError:
            print(f"Error: gcc not found. Please ensure gcc is installed and in your PATH.")
            # raise BuildFailed("gcc not found.")


setup(
    # name, version, install_requires, python_requires are defined in pyproject.toml
    # and will be used automatically by setuptools.
    author="Your Name / Original Author", # Placeholder - PLEASE UPDATE
    author_email="your.email@example.com", # Placeholder - PLEASE UPDATE
    description="3DDFA_V2 with dependencies packaged", # Placeholder - PLEASE UPDATE
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your_repo_path", # Placeholder - PLEASE UPDATE
    packages=find_packages(
        exclude=[
            'configs',
            'docs',
            'examples',
            'FaceBoxes.weights', # More specific exclusion
            'Sim3DR.tests',    # More specific exclusion
            'weights',
            'build',
            '*.tests', 
            '*.tests.*', 
            'tests.*', 
            'tests'
        ]
    ),
    ext_modules=ext_modules,
    # python_requires and install_requires are sourced from pyproject.toml
    include_package_data=True, # To include non-code files specified in MANIFEST.in (if any)
    package_data={
        'threeddfa.utils.asset': ['render.so'], # Ensure render.so is included
    },
    cmdclass={
        'build_py': BuildRenderSOCommand,
    },
    classifiers=[ # Placeholder classifiers - PLEASE UPDATE
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License", # Assuming MIT from LICENSE file
        "Operating System :: OS Independent",
    ],
    zip_safe=False, # Good practice for packages with C extensions
)