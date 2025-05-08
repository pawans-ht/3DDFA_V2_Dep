# coding: utf-8

__author__ = 'cleardusk'

import sys
import argparse
import cv2
import yaml

from .FaceBoxes import FaceBoxes
from .TDDFA import TDDFA
from .utils.render import render
#from .utils.render_ctypes import render  # faster
from .utils.depth import depth
from .utils.pncc import pncc
from .utils.uv import uv_tex
from .utils.pose import viz_pose
from .utils.serialization import ser_to_ply, ser_to_obj
from .utils.functions import draw_landmarks, get_suffix
from .utils.tddfa_util import str2bool


def main(args):
    # Ensure make_abs_path is defined in this script if it's not already
    # make_abs_path = lambda fn: osp.join(osp.dirname(osp.realpath(__file__)), fn)
    # It seems make_abs_path is not defined in demo.py, let's add it or assume config path is now relative to execution
    # For packaged apps, it's better to load package resources.
    # However, for demo scripts often run from package root or with adjusted CWD,
    # or if config path is passed explicitly, this might work.
    # Assuming args.config will be 'threeddfa/configs/xxx.yml' if run from project root,
    # or just 'xxx.yml' if the script itself handles pathing via make_abs_path.
    # The argparse default needs to be relative to where the script expects to find it.
    # If make_abs_path is defined as:
    # make_abs_path = lambda fn: osp.join(osp.dirname(osp.realpath(__file__)), fn)
    # then the default in argparse should be relative to this file.
    # Since configs are now in threeddfa/configs, and demo.py is in threeddfa/,
    # the relative path from demo.py to configs/ is just 'configs/'.
    
    # If args.config is a full path or relative path from CWD, open() is fine.
    # If args.config is just a filename, it needs to be resolved.
    # The original default='configs/mb1_120x120.yml' implies it was relative to project root.
    # Now, if the script is in threeddfa/, and configs in threeddfa/configs/,
    # the path from script to config is 'configs/filename.yml'
    
    # Let's assume args.config might be passed as a full path or a path relative to CWD.
    # If it's the default value, it needs to be correct.
    # The `make_abs_path` in other files is usually for files *next to* or *below* the current script.
    # Here, `configs` is a sibling directory to `demo.py` if `demo.py` is at `threeddfa/demo.py`
    # and configs are at `threeddfa/configs`.
    
    # The most robust way for default configs is to make them relative to the script file.
    # Add make_abs_path if not present
    if 'make_abs_path' not in globals():
        import os.path as osp
        make_abs_path = lambda fn: osp.join(osp.dirname(osp.realpath(__file__)), fn)
        
    config_path = args.config
    if not osp.isabs(config_path) and not config_path.startswith('configs/'):
        # If it's a simple filename from default, make it relative to script's dir then into 'configs'
         # This assumes the default in argparse is just 'mb1_120x120.yml'
         # and we want to find it in the 'configs' dir relative to this script.
        pass # The default path in argparse will be adjusted instead.

    cfg = yaml.load(open(args.config), Loader=yaml.SafeLoader)

    # Init FaceBoxes and TDDFA, recommend using onnx flag
    if args.onnx:
        import os
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        os.environ['OMP_NUM_THREADS'] = '4'

        from .FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
        from .TDDFA_ONNX import TDDFA_ONNX

        face_boxes = FaceBoxes_ONNX()
        tddfa = TDDFA_ONNX(**cfg)
    else:
        gpu_mode = args.mode == 'gpu'
        tddfa = TDDFA(gpu_mode=gpu_mode, **cfg)
        face_boxes = FaceBoxes()

    # Given a still image path and load to BGR channel
    img = cv2.imread(args.img_fp)

    # Detect faces, get 3DMM params and roi boxes
    boxes = face_boxes(img)
    n = len(boxes)
    if n == 0:
        print(f'No face detected, exit')
        sys.exit(-1)
    print(f'Detect {n} faces')

    param_lst, roi_box_lst = tddfa(img, boxes)

    # Visualization and serialization
    dense_flag = args.opt in ('2d_dense', '3d', 'depth', 'pncc', 'uv_tex', 'ply', 'obj')
    old_suffix = get_suffix(args.img_fp)
    new_suffix = f'.{args.opt}' if args.opt in ('ply', 'obj') else '.jpg'

    wfp = f'examples/results/{args.img_fp.split("/")[-1].replace(old_suffix, "")}_{args.opt}' + new_suffix

    ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)

    if args.opt == '2d_sparse':
        draw_landmarks(img, ver_lst, show_flag=args.show_flag, dense_flag=dense_flag, wfp=wfp)
    elif args.opt == '2d_dense':
        draw_landmarks(img, ver_lst, show_flag=args.show_flag, dense_flag=dense_flag, wfp=wfp)
    elif args.opt == '3d':
        render(img, ver_lst, tddfa.tri, alpha=0.6, show_flag=args.show_flag, wfp=wfp)
    elif args.opt == 'depth':
        # if `with_bf_flag` is False, the background is black
        depth(img, ver_lst, tddfa.tri, show_flag=args.show_flag, wfp=wfp, with_bg_flag=True)
    elif args.opt == 'pncc':
        pncc(img, ver_lst, tddfa.tri, show_flag=args.show_flag, wfp=wfp, with_bg_flag=True)
    elif args.opt == 'uv_tex':
        uv_tex(img, ver_lst, tddfa.tri, show_flag=args.show_flag, wfp=wfp)
    elif args.opt == 'pose':
        viz_pose(img, param_lst, ver_lst, show_flag=args.show_flag, wfp=wfp)
    elif args.opt == 'ply':
        ser_to_ply(ver_lst, tddfa.tri, height=img.shape[0], wfp=wfp)
    elif args.opt == 'obj':
        ser_to_obj(img, ver_lst, tddfa.tri, height=img.shape[0], wfp=wfp)
    else:
        raise ValueError(f'Unknown opt {args.opt}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The demo of still image of 3DDFA_V2')
    # make_abs_path for default config, assuming configs are in threeddfa/configs
    # and this script is in threeddfa/
    # So, path from this script to configs/mb1_120x120.yml is 'configs/mb1_120x120.yml'
    # We need to ensure make_abs_path is defined before parser, or use a simpler default.
    # For simplicity in argparse, let's make the default path directly usable if script is run from threeddfa directory
    # or if a full path is provided.
    # The default should be relative to where the script is, if __file__ is used.
    # If script is threeddfa/demo.py, default should be 'configs/mb1_120x120.yml'
    # to load threeddfa/configs/mb1_120x120.yml
    # Let's define make_abs_path at the top of the script for clarity.
    parser.add_argument('-c', '--config', type=str, default='configs/mb1_120x120.yml') # Path relative to threeddfa/
    parser.add_argument('-f', '--img_fp', type=str, default='examples/inputs/trump_hillary.jpg')
    parser.add_argument('-m', '--mode', type=str, default='cpu', help='gpu or cpu mode')
    parser.add_argument('-o', '--opt', type=str, default='2d_sparse',
                        choices=['2d_sparse', '2d_dense', '3d', 'depth', 'pncc', 'uv_tex', 'pose', 'ply', 'obj'])
    parser.add_argument('--show_flag', type=str2bool, default='true', help='whether to show the visualization result')
    parser.add_argument('--onnx', action='store_true', default=False)

    args = parser.parse_args()
    main(args)
