# coding: utf-8

__author__ = 'cleardusk'

import sys

# sys.path.append('..') # Removed

import os.path as osp
import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

from ..utils.io import _load, _numpy_to_cuda, _numpy_to_tensor

# make_abs_path = lambda fn: osp.join(osp.dirname(osp.realpath(__file__)), fn) # Not needed if loading from HF


def _to_ctype(arr):
    if not arr.flags.c_contiguous:
        return arr.copy(order='C')
    return arr


def _load_tri(bfm_filename_on_hub): # bfm_fp is now the filename on Hub
    HF_REPO_ID = "Stable-Human/3ddfa_v2"
    TRI_FILENAME = "tri.pkl"
    
    tri_data = None
    try:
        # For 'bfm_noneck_v3.pkl' (or its .onnx equivalent context), 'tri.pkl' is loaded separately.
        if bfm_filename_on_hub == 'bfm_noneck_v3.pkl' or bfm_filename_on_hub == 'bfm_noneck_v3.onnx':
            downloaded_tri_fp = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=TRI_FILENAME
            )
            tri_data = _load(downloaded_tri_fp)
        else:
            # For other BFM models, assume 'tri' is part of the main BFM file
            # This part might need adjustment if other BFM files are used and also need separate tri files.
            downloaded_bfm_fp = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=bfm_filename_on_hub
            )
            bfm_content = _load(downloaded_bfm_fp)
            tri_data = bfm_content.get('tri')
            
    except Exception as e:
        print(f"Error downloading triangle data for {bfm_filename_on_hub} from {HF_REPO_ID}: {e}")
        raise

    if tri_data is None:
        raise ValueError(f"Triangle data (tri) could not be loaded for {bfm_filename_on_hub}")
        
    tri = _to_ctype(tri_data.T).astype(np.int32)
    return tri


class BFMModel_ONNX(nn.Module):
    """BFM serves as a decoder"""

    def __init__(self, bfm_fp, shape_dim=40, exp_dim=10):
        super(BFMModel_ONNX, self).__init__()

        _to_tensor = _numpy_to_tensor

        # load bfm
        # bfm_fp is the filename on Hugging Face Hub
        HF_REPO_ID = "Stable-Human/3ddfa_v2"
        try:
            downloaded_bfm_fp = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=bfm_fp
            )
            bfm = _load(downloaded_bfm_fp)
        except Exception as e:
            print(f"Error downloading BFM model {bfm_fp} from {HF_REPO_ID}: {e}")
            raise

        u = _to_tensor(bfm.get('u').astype(np.float32))
        self.u = u.view(-1, 3).transpose(1, 0)
        w_shp = _to_tensor(bfm.get('w_shp').astype(np.float32)[..., :shape_dim])
        w_exp = _to_tensor(bfm.get('w_exp').astype(np.float32)[..., :exp_dim])
        w = torch.cat((w_shp, w_exp), dim=1)
        self.w = w.view(-1, 3, w.shape[-1]).contiguous().permute(1, 0, 2)

        # self.u = _to_tensor(bfm.get('u').astype(np.float32))  # fix bug
        # w_shp = _to_tensor(bfm.get('w_shp').astype(np.float32)[..., :shape_dim])
        # w_exp = _to_tensor(bfm.get('w_exp').astype(np.float32)[..., :exp_dim])
        # self.w = torch.cat((w_shp, w_exp), dim=1)

        # self.keypoints = bfm.get('keypoints').astype(np.long)  # fix bug
        # self.u_base = self.u[self.keypoints].reshape(-1, 1)
        # self.w_shp_base = self.w_shp[self.keypoints]
        # self.w_exp_base = self.w_exp[self.keypoints]

    def forward(self, *inps):
        R, offset, alpha_shp, alpha_exp = inps
        alpha = torch.cat((alpha_shp, alpha_exp))
        # pts3d = R @ (self.u + self.w_shp.matmul(alpha_shp) + self.w_exp.matmul(alpha_exp)). \
        #     view(-1, 3).transpose(1, 0) + offset
        # pts3d = R @ (self.u + self.w.matmul(alpha)).view(-1, 3).transpose(1, 0) + offset
        pts3d = R @ (self.u + self.w.matmul(alpha).squeeze()) + offset
        return pts3d


def convert_bfm_to_onnx(target_onnx_filename_on_hub, shape_dim=40, exp_dim=10):
    # This function converts a .pkl BFM model (fetched from Hub) to an ONNX model.
    # The resulting ONNX model is saved locally with the target_onnx_filename_on_hub name.
    # It's assumed this ONNX file would then be manually uploaded to the Hub.
    
    HF_REPO_ID = "Stable-Human/3ddfa_v2"
    # Assumes the .pkl filename can be derived from the target .onnx filename
    bfm_pkl_filename_on_hub = target_onnx_filename_on_hub.replace('.onnx', '.pkl')

    print(f"Attempting to convert {bfm_pkl_filename_on_hub} from Hub to local {target_onnx_filename_on_hub}")

    bfm_decoder = BFMModel_ONNX(bfm_fp=bfm_pkl_filename_on_hub, shape_dim=shape_dim, exp_dim=exp_dim)
    bfm_decoder.eval()

    # dummy_input = torch.randn(12 + shape_dim + exp_dim)
    dummy_input = torch.randn(3, 3), torch.randn(3, 1), torch.randn(shape_dim, 1), torch.randn(exp_dim, 1)
    R, offset, alpha_shp, alpha_exp = dummy_input
    torch.onnx.export(
        bfm_decoder,
        (R, offset, alpha_shp, alpha_exp),
        target_onnx_filename_on_hub, # Save locally with this name
        input_names=['R', 'offset', 'alpha_shp', 'alpha_exp'],
        output_names=['output'],
        dynamic_axes={
            'alpha_shp': [0],
            'alpha_exp': [0],
        },
        do_constant_folding=True
    )
    print(f'Convert {bfm_pkl_filename_on_hub} (from Hub) to local {target_onnx_filename_on_hub} done.')


if __name__ == '__main__':
    # This main block is for development/testing to generate the ONNX file.
    # The generated file 'bfm_noneck_v3.onnx' should then be uploaded to Hugging Face Hub.
    # The function convert_bfm_to_onnx now expects the target ONNX filename as it would appear on the Hub (or locally).
    # It derives the .pkl filename from it to download from the Hub.
    target_local_bfm_onnx_filename = "bfm_noneck_v3.onnx"
    print(f"Running conversion for {target_local_bfm_onnx_filename}. Output will be local.")
    print("Ensure the corresponding .pkl file (e.g., bfm_noneck_v3.pkl) is on Hugging Face Hub.")
    convert_bfm_to_onnx(target_local_bfm_onnx_filename)
