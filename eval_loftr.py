import cv2
import kornia as K
import kornia.feature as KF
import numpy as np
import torch
from kornia_moons.feature import *

def load_torch_image(img):
    img = K.image_to_tensor(img, False).float()/255.
    img = K.color.bgr_to_rgb(img)
    return img.cuda()

class LoFTRMatcher():
    def __init__(self):
        self.matcher = KF.LoFTR(pretrained='outdoor').cuda()
        
    def match(self, img1, img2):
        if img1.shape[-2] == 3:
            img1 = load_torch_image(img1)
            img2 = load_torch_image(img2)
            input_dict = {"image0": K.color.rgb_to_grayscale(img1), # LofTR works on grayscale images only 
                        "image1": K.color.rgb_to_grayscale(img2)}
        else:
            input_dict = {"image0": (K.image_to_tensor(img1, False).float() /255.).cuda(), # LofTR works on grayscale images only 
                        "image1": (K.image_to_tensor(img2, False).float() /255.).cuda()}
        with torch.inference_mode():
            correspondences = self.matcher(input_dict)    
            mkpts0 = correspondences['keypoints0'].cpu().numpy()
            mkpts1 = correspondences['keypoints1'].cpu().numpy()
            # Fm, inliers = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.5, 0.999, 100000)
            # inliers = inliers > 0
        return mkpts0, mkpts1


from eval_megadepth_scannet import evaluate_megadepth, evaluate_scannet

loftr = LoFTRMatcher()

# evaluate_megadepth(loftr.match)

evaluate_scannet(loftr.match)


