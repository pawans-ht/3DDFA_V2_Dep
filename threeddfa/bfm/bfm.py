# coding: utf-8

__author__ = 'cleardusk'

import sys

# sys.path.append('..') # Removed

import os.path as osp
import numpy as np
from ..utils.io import _load
from huggingface_hub import hf_hub_download

# make_abs_path = lambda fn: osp.join(osp.dirname(osp.realpath(__file__)), fn) # Not needed if loading from HF


def _to_ctype(arr):
    if not arr.flags.c_contiguous:
        return arr.copy(order='C')
    return arr


class BFMModel(object):
    def __init__(self, bfm_fp_placeholder=None, shape_dim=40, exp_dim=10): # bfm_fp is now a placeholder or filename
        HF_REPO_ID = "Stable-Human/3ddfa_v2"
        BFM_MODEL_FILENAME = "bfm_noneck_v3.pkl" # Assuming this is the primary BFM file
        TRI_FILENAME = "tri.pkl"

        # Determine BFM model filename to download
        # If bfm_fp_placeholder is provided and is a filename, use it. Otherwise, use default.
        actual_bfm_filename = BFM_MODEL_FILENAME
        if bfm_fp_placeholder and not osp.isdir(bfm_fp_placeholder): # if it's a filename, not a path
             actual_bfm_filename = osp.basename(bfm_fp_placeholder)


        try:
            downloaded_bfm_fp = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=actual_bfm_filename
            )
            bfm = _load(downloaded_bfm_fp)
            
            # Determine if we need to download a separate tri.pkl or if it's in the BFM_MODEL_FILENAME
            # For 'bfm_noneck_v3.pkl', the original code loaded 'tri.pkl' separately.
            if actual_bfm_filename == "bfm_noneck_v3.pkl":
                 downloaded_tri_fp = hf_hub_download(
                    repo_id=HF_REPO_ID,
                    filename=TRI_FILENAME
                )
                 self.tri = _load(downloaded_tri_fp)
            else:
                self.tri = bfm.get('tri') # Assumes tri is in other bfm files

        except Exception as e:
            print(f"Error downloading BFM assets from {HF_REPO_ID}: {e}")
            raise

        self.u = bfm.get('u').astype(np.float32)
        self.w_shp = bfm.get('w_shp').astype(np.float32)[..., :shape_dim]
        self.w_exp = bfm.get('w_exp').astype(np.float32)[..., :exp_dim]
        
        if self.tri is None: # Should not happen if logic above is correct
            raise ValueError("Triangle data (tri) could not be loaded.")
            
        self.tri = _to_ctype(self.tri.T).astype(np.int32)
        self.keypoints = bfm.get('keypoints').astype(np.long)  # fix bug
        w = np.concatenate((self.w_shp, self.w_exp), axis=1)
        self.w_norm = np.linalg.norm(w, axis=0)

        self.u_base = self.u[self.keypoints].reshape(-1, 1)
        self.w_shp_base = self.w_shp[self.keypoints]
        self.w_exp_base = self.w_exp[self.keypoints]
