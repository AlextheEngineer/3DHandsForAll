import os
import cv2
import math
import torch
import numpy as np
import params

def load_img_mano(img_path):
    img_orig = cv2.imread(img_path)
    img_h, img_w = img_orig.shape[:2]
    long_side = max(img_h, img_w)
    scale = params.IMG_SIZE/long_side
    img_resized_h, img_resized_w = round(img_h*scale), round(img_w*scale)
    img_resized = cv2.resize(img_orig, (img_resized_w, img_resized_h))
    padding_top, padding_bot, padding_left, padding_right = 0, 0, 0, 0
    if img_resized_w > img_resized_h:
        # Pad top and bot
        padding_top = round((params.IMG_SIZE - img_resized_h)/2)
        padding_bot = params.IMG_SIZE - img_resized_h - padding_top
    elif img_resized_w < img_resized_h:
        # Pad left and right
        padding_left = round((params.IMG_SIZE - img_resized_w)/2)
        padding_right = params.IMG_SIZE - img_resized_w - padding_left
    img_padded = cv2.copyMakeBorder(img_resized, top=padding_top, bottom=padding_bot, left=padding_left, \
        right=padding_right, borderType=cv2.BORDER_CONSTANT)
    # For tkinter display
    img_padded = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
    return img_orig, img_padded
    
def paint_kpts(img_path, img, kpts, circle_size = 1):
    colors = params.colors
    # To be continued
    limbSeq = params.limbSeq_hand
    
    im = cv2.imread(img_path) if img is None else img.copy()
    # draw points
    for k, kpt in enumerate(kpts):
        row = int(kpt[0])
        col = int(kpt[1])
        if k in [0, 4, 8, 12, 16, 20]:
            r = circle_size
        else:
            r = 1
        cv2.circle(im, (col, row), radius=r, thickness=-1, color=(0, 0, 255))

    # draw lines
    for i in range(len(limbSeq)):
        cur_im = im.copy()
        limb = limbSeq[i]
        [X0, Y0] = kpts[limb[0]]
        [X1, Y1] = kpts[limb[1]]
        mX = np.mean([X0, X1])
        mY = np.mean([Y0, Y1])
        length = ((X0 - X1) ** 2 + (Y0 - Y1) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X0 - X1, Y0 - Y1))
        polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), 4), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(cur_im, polygon, colors[i])
        im = cv2.addWeighted(im, 0.4, cur_im, 0.6, 0)

    return im
    
def gaussian_kernel(size_w, size_h, center_x, center_y, sigma, z = -2.0):
    gridy, gridx = np.mgrid[0:size_h, 0:size_w]
    D2 = (gridx - center_x) ** 2 + (gridy - center_y) ** 2
    spread = 1.0
    return np.exp(-D2 / 2.0 / sigma / sigma * spread)
    
def generate_heatmaps(img_shape, stride, kpt_2d, sigma=2.0, is_ratio=False):
    kpt_2d = kpt_2d.copy()
        
    height, width = img_shape[:2]
    if is_ratio:
        kpt_2d *= np.array([height, width])

    heatmaps = np.zeros((int(height / stride), int(width / stride), len(kpt_2d) + 1), dtype=np.float32)
    #sigma = 3.0
    for i in range(len(kpt_2d)):
        y = int(kpt_2d[i][0]) * 1.0 / stride
        x = int(kpt_2d[i][1]) * 1.0 / stride
        z = None
        heat_map = gaussian_kernel(size_h=height / stride, size_w=width / stride, center_x=x, \
            center_y=y, sigma=sigma, z = z)
        heat_map[heat_map > 1] = 1
        heat_map[heat_map < 0.0099] = 0
        heatmaps[:, :, i + 1] = heat_map

    heatmaps[:, :, 0] = 1.0 - np.max(heatmaps[:, :, 1:], axis=2)  # for background
    return heatmaps
    
def normalize_tensor(tensor, mean, std):
    for t in tensor:
        t.sub_(mean).div_(std)
    return tensor
    
def get_2d_kpts(heatmaps, img_h, img_w, num_keypoints):
    kpts = np.zeros((num_keypoints, 2))
    for idx, m in enumerate(heatmaps[1:]):
        h, w = np.unravel_index(m.argmax(), m.shape)
        col = int(w * img_w / m.shape[1])
        row = int(h * img_h / m.shape[0])
        kpts[idx] = np.array([row, col])
    
    return kpts
    
def clip_mano_hand_rot(rot_tensor):
    rot_min_tensor = torch.tensor([
        -10.0, -10.0, -10.0
        ]).cuda()
        
    rot_max_tensor = torch.tensor([
        10.0, 10.0, 10.0
        ]).cuda()
    return torch.min(torch.max(rot_tensor, rot_min_tensor), rot_max_tensor)
    
def clip_mano_hand_pose(pose_tensor):
    pose_min_tensor = torch.tensor(params.rot_min_list[3:]).cuda()
    pose_max_tensor = torch.tensor(params.rot_max_list[3:]).cuda()
    return torch.min(torch.max(pose_tensor, pose_min_tensor), pose_max_tensor)
    
def clip_mano_hand_shape(shape_tensor):
    shape_min_tensor = torch.tensor([
        -10.0, -10.0, -10.0, -10.0, -10.0,
        -10.0, -10.0, -10.0, -10.0, -10.0
    ]).cuda()
    shape_max_tensor = torch.tensor([
        10.0, 10.0, 10.0, 10.0, 10.0,
        10.0, 10.0, 10.0, 10.0, 10.0
    ]).cuda()
    return torch.min(torch.max(shape_tensor, shape_min_tensor), shape_max_tensor)
    
def leapdata_scaled2ego(leap, ego):
    leap = leap.detach().cpu().numpy()
    ego = ego.detach().cpu().numpy()

    # Resulting root joint should always be the original leap point.
    res = np.zeros(leap.shape)
    res[0, :] = leap[0, :]
    
    for finger in params.EGO_HAND_INDICES:
        prev_idx = 0
        shift_total = np.zeros((3))
        for curr_idx in params.EGO_HAND_INDICES[finger]:
            ego_dist = np.linalg.norm(ego[curr_idx] - ego[prev_idx])

            leap_diff = leap[curr_idx] - leap[prev_idx]
            leap_dist = np.linalg.norm(leap_diff)
            leap_dir = leap_diff / leap_dist

            # Joint lengths are not always consistent between ego and leap
            shift_dist = ego_dist - leap_dist
            shift_dir = leap_dir
            shift_total += shift_dist * shift_dir

            res[curr_idx, :] = (leap[curr_idx] + shift_total)[:]
            prev_idx = curr_idx

    return torch.from_numpy(res).cuda()
    
def get_pose_constraint_tensor():
    pose_mean_tensor = torch.tensor(params.pose_mean_list).cuda()
    pose_reg_tensor = torch.tensor(params.pose_reg_list).cuda()
    return pose_reg_tensor, pose_mean_tensor
    
def find_nearest_idx_in_array(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx
    
def convert_img2mano(img, img_shape_mano):
    img_shape_orig = img.shape[:2]
    top_padding = int((img_shape_orig[1] - img_shape_orig[0])/2)
    bot_padding = img_shape_orig[1] - top_padding - img_shape_orig[0]
    img = cv2.copyMakeBorder(img, top_padding, bot_padding, 0, 0, cv2.BORDER_CONSTANT, value = 0)
    return img
    
def convert_mano2img(img_mano, img_shape):
    img_shape_mano = img_mano.shape[:2]
    row_start = int((img_shape_mano[0] - img_shape[0])/2)
    row_end = row_start + img_shape[0]
    return img_mano[row_start:row_end, :, :]
       
# UI functions
def get_slider_from_mano_val(mano_id, rot_idx, mano_val):
    mano_val = float(mano_val)
    if mano_id == "xyz_root":
        return find_nearest_idx_in_array(params.mano_xyz_root_range_list[rot_idx], mano_val)
    elif mano_id == "shape":
        return find_nearest_idx_in_array(params.mano_shape_range_list[rot_idx], mano_val)
    elif mano_id == "glob":
        return find_nearest_idx_in_array(params.mano_glob_rot_range_list[rot_idx], mano_val)
    elif mano_id == "finger0":
        return find_nearest_idx_in_array(params.mano_finger0_rot_range_list[rot_idx], mano_val)
    elif mano_id == "finger1":
        return find_nearest_idx_in_array(params.mano_finger1_rot_range_list[rot_idx], mano_val)
    elif mano_id == "finger2":
        return find_nearest_idx_in_array(params.mano_finger2_rot_range_list[rot_idx], mano_val)
    elif mano_id == "finger3":
        return find_nearest_idx_in_array(params.mano_finger3_rot_range_list[rot_idx], mano_val)
    elif mano_id == "finger4":
        return find_nearest_idx_in_array(params.mano_finger4_rot_range_list[rot_idx], mano_val)
        
def get_mano_val_from_slider(mano_id, rot_idx, tick_idx):
    if mano_id == "xyz_root":
        return params.mano_xyz_root_range_list[rot_idx][tick_idx]
    elif mano_id == "shape":
        return params.mano_shape_range_list[rot_idx][tick_idx]
    elif mano_id == "glob":
        return params.mano_glob_rot_range_list[rot_idx][tick_idx]
    elif mano_id == "finger0":
        return params.mano_finger0_rot_range_list[rot_idx][tick_idx]
    elif mano_id == "finger1":
        return params.mano_finger1_rot_range_list[rot_idx][tick_idx]
    elif mano_id == "finger2":
        return params.mano_finger2_rot_range_list[rot_idx][tick_idx]
    elif mano_id == "finger3":
        return params.mano_finger3_rot_range_list[rot_idx][tick_idx]
    elif mano_id == "finger4":
        return params.mano_finger4_rot_range_list[rot_idx][tick_idx]
        
def assert_valid_crop(crop_info):
    img_crop_top, img_crop_bot, img_crop_left, img_crop_right, padding_top, padding_bot, \
        padding_left, padding_right, crop_h, crop_w = crop_info
    assert crop_h == crop_w, "Invalid crop with height of {} and width of {}".format(crop_h, crop_w)
    
def get_crop_attr_from_kpts_2d(kpts_2d_glob):
    kpts_2d_mean = np.rint(np.mean(kpts_2d_glob, 0)).astype(np.int32)
    crop_center = (kpts_2d_mean[0], kpts_2d_mean[1])
    max_dist = max(np.max(kpts_2d_glob[:, 0]) - np.min(kpts_2d_glob[:, 0]), \
        np.max(kpts_2d_glob[:, 1]) - np.min(kpts_2d_glob[:, 1]))
    crop_size = round(max_dist * 1.5)
    return crop_center, crop_size