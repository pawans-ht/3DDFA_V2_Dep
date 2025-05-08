# coding: utf-8

__author__ = 'cleardusk'

import os.path as osp
import numpy as np
import cv2
import onnxruntime
from huggingface_hub import hf_hub_download

from .utils.onnx import convert_to_onnx
from .utils.io import _load
from .utils.functions import (
    crop_img, parse_roi_box_from_bbox, parse_roi_box_from_landmark,
)
from .utils.tddfa_util import _parse_param, similar_transform
from .bfm.bfm import BFMModel
from .bfm.bfm_onnx import convert_bfm_to_onnx

make_abs_path = lambda fn: osp.join(osp.dirname(osp.realpath(__file__)), fn)


class TDDFA_ONNX(object):
    """TDDFA_ONNX: the ONNX version of Three-D Dense Face Alignment (TDDFA)"""

    def __init__(self, **kvs):
        # torch.set_grad_enabled(False)

        # load onnx version of BFM
        bfm_fp = kvs.get('bfm_fp', make_abs_path('../configs/bfm_noneck_v3.pkl')) # Adjusted path
        bfm_onnx_fp = bfm_fp.replace('.pkl', '.onnx')
        if not osp.exists(bfm_onnx_fp):
            convert_bfm_to_onnx(
                bfm_onnx_fp,
                shape_dim=kvs.get('shape_dim', 40),
                exp_dim=kvs.get('exp_dim', 10)
            )
        self.bfm_session = onnxruntime.InferenceSession(bfm_onnx_fp, None)

        # load for optimization
        bfm = BFMModel(bfm_fp, shape_dim=kvs.get('shape_dim', 40), exp_dim=kvs.get('exp_dim', 10))
        self.tri = bfm.tri
        self.u_base, self.w_shp_base, self.w_exp_base = bfm.u_base, bfm.w_shp_base, bfm.w_exp_base

        # config
        self.gpu_mode = kvs.get('gpu_mode', False)
        self.gpu_id = kvs.get('gpu_id', 0)
        self.size = kvs.get('size', 120)

        param_mean_std_fp = kvs.get(
            'param_mean_std_fp', make_abs_path(f'../configs/param_mean_std_62d_{self.size}x{self.size}.pkl') # Adjusted path
        )

        # Determine the ONNX model filename expected on Hugging Face Hub
        # This assumes kvs['checkpoint_fp'] is the original .pth filename (e.g., "mb1_120x120.pth")
        # and kvs['onnx_fp'] if provided is the direct filename (e.g., "mb1_120x120.onnx")
        onnx_filename_on_hub = kvs.get('onnx_fp')
        if not onnx_filename_on_hub and kvs.get('checkpoint_fp'):
            onnx_filename_on_hub = kvs.get('checkpoint_fp').replace('.pth', '.onnx')
        elif not onnx_filename_on_hub:
            # Fallback or error if no way to determine ONNX filename
            raise ValueError("ONNX filename cannot be determined. Provide 'onnx_fp' or 'checkpoint_fp'.")

        # --- Define your Hugging Face Hub details ---
        HF_REPO_ID = "Stable-Human/3ddfa_v2" # TODO: REPLACE THIS
        # ---

        try:
            downloaded_onnx_fp = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=onnx_filename_on_hub,
            )
            self.session = onnxruntime.InferenceSession(downloaded_onnx_fp, None)
        except Exception as e:
            # Original code had a fallback to convert .pth to .onnx.
            # For simplicity with Hugging Face Hub, it's best to upload pre-converted .onnx files.
            # If on-the-fly conversion is essential:
            # 1. Download the .pth file using hf_hub_download.
            # 2. Call a modified convert_to_onnx (from .utils.onnx) that takes the .pth path,
            #    converts it, saves to a local cache, and returns the cached .onnx path.
            # 3. Load the session from the cached .onnx path.
            # This is more complex and not implemented here.
            print(f"Error downloading/loading ONNX model {onnx_filename_on_hub} from {HF_REPO_ID}: {e}")
            print("Please ensure the ONNX model is available on Hugging Face Hub or implement on-the-fly conversion.")
            raise

        # params normalization config
        r = _load(param_mean_std_fp)
        self.param_mean = r.get('mean')
        self.param_std = r.get('std')

    def __call__(self, img_ori, objs, **kvs):
        # Crop image, forward to get the param
        param_lst = []
        roi_box_lst = []

        crop_policy = kvs.get('crop_policy', 'box')
        for obj in objs:
            if crop_policy == 'box':
                # by face box
                roi_box = parse_roi_box_from_bbox(obj)
            elif crop_policy == 'landmark':
                # by landmarks
                roi_box = parse_roi_box_from_landmark(obj)
            else:
                raise ValueError(f'Unknown crop policy {crop_policy}')

            roi_box_lst.append(roi_box)
            img = crop_img(img_ori, roi_box)
            img = cv2.resize(img, dsize=(self.size, self.size), interpolation=cv2.INTER_LINEAR)
            img = img.astype(np.float32).transpose(2, 0, 1)[np.newaxis, ...]
            img = (img - 127.5) / 128.

            inp_dct = {'input': img}

            param = self.session.run(None, inp_dct)[0]
            param = param.flatten().astype(np.float32)
            param = param * self.param_std + self.param_mean  # re-scale
            param_lst.append(param)

        return param_lst, roi_box_lst

    def recon_vers(self, param_lst, roi_box_lst, **kvs):
        dense_flag = kvs.get('dense_flag', False)
        size = self.size

        ver_lst = []
        for param, roi_box in zip(param_lst, roi_box_lst):
            R, offset, alpha_shp, alpha_exp = _parse_param(param)
            if dense_flag:
                inp_dct = {
                    'R': R, 'offset': offset, 'alpha_shp': alpha_shp, 'alpha_exp': alpha_exp
                }
                pts3d = self.bfm_session.run(None, inp_dct)[0]
                pts3d = similar_transform(pts3d, roi_box, size)
            else:
                pts3d = R @ (self.u_base + self.w_shp_base @ alpha_shp + self.w_exp_base @ alpha_exp). \
                    reshape(3, -1, order='F') + offset
                pts3d = similar_transform(pts3d, roi_box, size)

            ver_lst.append(pts3d)

        return ver_lst
