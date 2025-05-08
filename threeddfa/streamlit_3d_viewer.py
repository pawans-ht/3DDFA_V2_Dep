import streamlit as st
import cv2
import yaml
import numpy as np
import os.path as osp
from PIL import Image

# Assuming the script is run from the root of the 3DDFA_V2_Dep project
# Adjust paths if necessary or ensure 3DDFA is in PYTHONPATH
from threeddfa.FaceBoxes import FaceBoxes
from threeddfa.TDDFA import TDDFA
from threeddfa.utils.tddfa_util import str2bool
from threeddfa.utils.functions import get_suffix # May not be needed for streamlit version
# For ONNX version if preferred later
# from threeddfa.FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
# from threeddfa.TDDFA_ONNX import TDDFA_ONNX

import plotly.graph_objects as go

# Define make_abs_path relative to the 3DDFA package if needed, or use absolute paths for configs
# For simplicity, we might hardcode or make config path selectable
CONFIG_PATH = 'threeddfa/configs/mb1_120x120.yml' # Default config

@st.cache_resource # Cache the model loading
def load_tddfa_model(config_path):
    """Loads the TDDFA model."""
    cfg = yaml.load(open(config_path), Loader=yaml.SafeLoader)
    # For now, let's stick to the standard version, not ONNX, for simplicity
    tddfa = TDDFA(gpu_mode=False, **cfg) # Assuming CPU mode for broader compatibility
    face_boxes = FaceBoxes()
    return tddfa, face_boxes

def main():
    st.set_page_config(layout="wide")
    st.title("3DDFA Interactive Landmark Viewer")

    tddfa, face_boxes = load_tddfa_model(CONFIG_PATH)

    st.sidebar.header("Controls")
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        pil_image = Image.open(uploaded_file)
        img_np = np.array(pil_image)
        img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR) # OpenCV uses BGR

        st.image(pil_image, caption="Uploaded Image", use_column_width=True, width=300)

        if st.sidebar.button("Process Image and Show 3D Landmarks"):
            with st.spinner("Detecting faces and extracting landmarks..."):
                boxes = face_boxes(img_cv2)
                if not boxes:
                    st.warning("No faces detected in the image.")
                    return

                param_lst, roi_box_lst = tddfa(img_cv2, boxes)
                # Using dense_flag=True for dense landmarks
                ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=True)

                if not ver_lst:
                    st.error("Could not reconstruct vertices.")
                    return

                # For simplicity, display landmarks for the first detected face
                landmarks_3d = ver_lst[0].T  # Transpose to (N_points, 3)
                
                # Create Plotly 3D scatter plot
                fig = go.Figure(data=[go.Scatter3d(
                    x=landmarks_3d[:, 0],
                    y=landmarks_3d[:, 1],
                    z=landmarks_3d[:, 2],
                    mode='markers',
                    marker=dict(
                        size=2,
                        color=landmarks_3d[:, 2], # Color by Z-coordinate
                        colorscale='Viridis',
                        opacity=0.8
                    ),
                    # hoverinfo='text', # Default hoverinfo is usually good
                    customdata=np.arange(landmarks_3d.shape[0]), # Pass indices as customdata
                    hovertemplate='<b>Index: %{customdata}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>'
                )])
                
                fig.update_layout(
                    title="Dense 3D Facial Landmarks",
                    margin=dict(l=0, r=0, b=0, t=40),
                    scene=dict(
                        xaxis_title='X',
                        yaxis_title='Y',
                        zaxis_title='Z',
                        aspectmode='data' # Makes axes scale to data
                    )
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.success(f"Processed {len(boxes)} face(s). Displaying landmarks for the first face.")
                st.info("Hover over points in the plot to see their index and coordinates.")

if __name__ == '__main__':
    main()