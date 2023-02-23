
import os
from tokenize import Double
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from megadepth import MegaDepthDataset
from scannet import ScanNetDataset
from metrics import compute_pose_errors, compute_symmetrical_epipolar_errors, convert_points_to_homogeneous, aggregate_metrics

from pose_eval_utils import (compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc, read_image,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics)


def get_scannet_loader():
    data_root_dir = 'data/scannet/test'
    TEST_BASE_PATH = "assets/scannet_test_1500"
    scene_list_path = f"{TEST_BASE_PATH}/scannet_test.txt"
    npz_dir = f"{TEST_BASE_PATH}"
    intrinsic_path = f"{TEST_BASE_PATH}/intrinsics.npz"
    
    with open(scene_list_path, 'r') as f:
        npz_names = [name.split()[0] for name in f.readlines()]

    print(npz_names)

    sets = []
    for npz_name in tqdm(npz_names):
        # `ScanNetDataset`/`MegaDepthDataset` load all data from npz_path when initialized, which might take time.
        npz_path = os.path.join(npz_dir, npz_name)
        sets.append(ScanNetDataset(root_dir=data_root_dir, npz_path=npz_path, intrinsic_path=intrinsic_path, mode='test', min_overlap_score=0.0))

    print(sets)

    datas = DataLoader(sets[0], batch_size=1, shuffle=True, drop_last=False, num_workers=2)

    print(datas.__len__())

    return datas


def get_megadepth_loader():
    data_root_dir = 'data/megadepth/test'
    TEST_BASE_PATH = "assets/megadepth_test_1500_scene_info"
    scene_list_path = f"{TEST_BASE_PATH}/megadepth_test_1500.txt"
    npz_dir = f"{TEST_BASE_PATH}"
    with open(scene_list_path, 'r') as f:
        npz_names = [name.split()[0] for name in f.readlines()]
        npz_names = [f'{n}.npz' for n in npz_names]
    
    sets = []
    for npz_name in tqdm(npz_names):
        # `ScanNetDataset`/`MegaDepthDataset` load all data from npz_path when initialized, which might take time.
        npz_path = os.path.join(npz_dir, npz_name)
        # sets.append(MegaDepthDataset(root_dir=data_root_dir, npz_path=npz_path, mode='test', min_overlap_score=0.0, img_resize=840, img_padding=True, depth_padding=True, df=8))
        sets.append(MegaDepthDataset(root_dir=data_root_dir, npz_path=npz_path, mode='test', min_overlap_score=0.0, img_resize=1200, img_padding=False, depth_padding=False, df=8))

    datas = DataLoader(sets[0]+sets[1]+sets[2]+sets[3]+sets[4], batch_size=1, shuffle=True, drop_last=False, num_workers=2)

    return datas



from itertools import chain
def flattenList(x):
    return list(chain(*x))

import numpy as np
import cv2



def calculate_one_pair(mkpts0, mkpts1, T_0to1, K0, K1, kpts0):
    epi_errs = compute_epipolar_error(mkpts0, mkpts1, T_0to1, K0, K1)
    correct = epi_errs < 5e-4
    num_correct = np.sum(correct)
    precision = np.mean(correct) if len(correct) > 0 else 0
    matching_score = num_correct / len(kpts0) if len(kpts0) > 0 else 0

    thresh = 1.  # In pixels relative to resized image size.
    ret = estimate_pose(mkpts0, mkpts1, K0, K1, thresh)
    if ret is None:
        err_t, err_R = np.inf, np.inf
    else:
        R, t, inliers = ret
        err_t, err_R = compute_pose_error(T_0to1, R, t)
    return err_R, err_t, precision, matching_score


def evaluate_megadepth(matcher=None):
    datas = get_megadepth_loader()
    pose_errors = []
    precisions = []
    matching_scores = []
    for i, data in enumerate(tqdm(datas)):
        s0 = data['scale0']
        s1 = data['scale1']
        
        img0 = data['image0'].squeeze(0).cpu().numpy()
        img1 = data['image1'].squeeze(0).cpu().numpy()

        mkpts0, mkpts1 = matcher(img0, img1)
        
        # print(mkpts0.shape)

        mkpts0[:, 0] = s0[0, 0] * mkpts0[:, 0]
        mkpts0[:, 1] = s0[0, 1] * mkpts0[:, 1]
        mkpts1[:, 0] = s1[0, 0] * mkpts1[:, 0]
        mkpts1[:, 1] = s1[0, 1] * mkpts1[:, 1]

        T_0to1 = data['T_0to1'][0].cpu().numpy()
        K0 = data['K0'][0].cpu().numpy()
        K1 = data['K1'][0].cpu().numpy()

        err_R, err_t, precision, matching_score = calculate_one_pair(mkpts0,
            mkpts1, T_0to1, K0, K1, mkpts0)
        pose_error = np.maximum(err_t, err_R)
        pose_errors.append(pose_error)
        precisions.append(precision)
        matching_scores.append(matching_score)
    
        # print(matching_score)

    thresholds = [5, 10, 20]
    aucs = pose_auc(pose_errors, thresholds)
    aucs = [100.*yy for yy in aucs]
    prec = 100.*np.mean(precisions)
    ms = 100.*np.mean(matching_scores)
    print('Evaluation Results (mean over {} pairs):'.format(len(datas)))
    print('AUC@5\t AUC@10\t AUC@20\t Prec\t MScore\t')
    print('{:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t'.format(
        aucs[0], aucs[1], aucs[2], prec, ms))





def evaluate_scannet(matcher=None):
    datas = get_scannet_loader()
    pose_errors = []
    precisions = []
    matching_scores = []
    for i, data in enumerate(tqdm(datas)):
        # s0 = data['scale0']
        # s1 = data['scale1']

        img0 = data['image0'].squeeze(0).squeeze(0).cpu().numpy()
        img1 = data['image1'].squeeze(0).squeeze(0).cpu().numpy()

        mkpts0, mkpts1 = matcher(img0, img1)

        T_0to1 = data['T_0to1'][0].cpu().numpy()
        K0 = data['K0'][0].cpu().numpy()
        K1 = data['K1'][0].cpu().numpy()

        err_R, err_t, precision, matching_score = calculate_one_pair(mkpts0,
            mkpts1, T_0to1, K0, K1, mkpts0)
        pose_error = np.maximum(err_t, err_R)
        pose_errors.append(pose_error)
        precisions.append(precision)
        matching_scores.append(matching_score)

    thresholds = [5, 10, 20]
    aucs = pose_auc(pose_errors, thresholds)
    aucs = [100.*yy for yy in aucs]
    prec = 100.*np.mean(precisions)
    ms = 100.*np.mean(matching_scores)
    print('Evaluation Results (mean over {} pairs):'.format(len(datas)))
    print('AUC@5\t AUC@10\t AUC@20\t Prec\t MScore\t')
    print('{:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t'.format(
        aucs[0], aucs[1], aucs[2], prec, ms))


if __name__ == '__main__':

    pass

    # evaluate_megadepth()

    # evaluate_scannet()

