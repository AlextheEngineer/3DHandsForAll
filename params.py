import numpy as np

MANO_PATH = 'models/MANO_layer/manopth/mano/models'
MODEL_2D_PATH = 'models_saved/omnihands_HRNet_2d_pretrained.pth.tar'
MODEL_3D_3RD_PATH = 'models_saved/mano3dhands_3rd_person_resnet_pose_3d_pretrained.pth.tar'
MODEL_3D_EGO_PATH = 'models_saved/mano3dhands_egocentric_resnet_pose_3d_pretrained.pth.tar'

UI_HEIGHT = 1000
IMG_SIZE = 400
IMG_PADDING = 10
SCALE_PAD = 0
BUTTON_X_PAD = 10
BUTTON_H = 5 # Controls the height of buttons
BUTTON_W = 10
FREEZE_TEXT = 'freeze'
RESTORE_TEXT = 'restore'
NUM_KPTS = 21

NUM_ROT_TICKS = 100
CROP_SIZE_DEFAULT = 150
CROP_SIZE_PRED = 224
CROP_STRIDE_PRED = 4
HEATMAP_SIGMA = 4.0
CROP_SIZE_TICK_SIZE = 10
CROP_MAX_SIZE = IMG_SIZE
CROP_MIN_SIZE = 100

limbSeq_hand = [
    [0, 1],		#Thumb1
    [1, 2],		#Thumb2
    [2, 3],		#Thumb3
    [3, 4],		#Thumb4

    [0, 5],		#index1
    [5, 6],		#index2
    [6, 7],		#index3
    [7, 8],		#index4

    [0, 9],		#middle1
    [9, 10],	#middle2
    [10 ,11],	#middle3
    [11, 12],	#middle4

    [0, 13],	#ring1
    [13, 14],	#ring2
    [14, 15],	#ring3
    [15, 16],	#ring4

    [0, 17],	#pinky1
    [17, 18],	#pinky2
    [18, 19],	#pinky3
    [19, 20]	#pinky4
]

colors = [
    [0, 255, 0],		#0
    [0, 223, 0],		#1
    [0, 191, 0],		#2
    [0, 159, 0],		#3

    [159, 255, 0],		#4
    [159, 223, 0],		#5
    [159, 191, 0],		#6
    [159, 159, 0],		#7

    [255, 0, 0], 		#8
    [223, 0, 0],		#9
    [191, 0, 0],	 	#10
    [159, 0, 0], 		#11

    [255, 0, 255],	 	#12
    [255, 0, 223],		#13
    [255, 0, 191],	 	#14
    [255, 0, 159],		#15

    [0, 0, 255], 		#16
    [0, 0, 223], 		#17
    [0, 0, 191], 		#18
    [0, 0, 159],		#19
]

EGO_HAND_INDICES = {
    'thumb': [1, 2, 3, 4],
    'index': [5, 6, 7, 8],
    'middle': [9, 10, 11, 12],
    'ring': [13, 14, 15, 16],
    'pinky': [17, 18, 19, 20],
}

# MANO wrapper params
hand_root_joint = 9
mano_key_bone_len = 10.0
pytorch3d_img_w = IMG_SIZE
lr_rot_init = 1.0
lr_pose_init = 0.1#0.01
lr_xyz_root_init = 1.0
lr_all_init = 0.4#0.01
lr_anno_2d = 0.1
lr_anno_xyz_root = 1.0

glob_rot_range = 6.28
rot_min_list = [
    -glob_rot_range, -glob_rot_range, -glob_rot_range,                  #0
    -0.25, -0.5, -1.0,                                                  #1
    0.0, 0.0, -1.0,                                                     #2
    0.0, 0.0, -0.3,                                                     #3
    -0.25, -0.5, -1.0,                                                  #4
    0.0, 0.0, -1.0,                                                     #5
    0.0, 0.0, -0.2,                                                     #6
    -0.25, -0.5, -1.25,                                                 #7
    -1.0, 0.0, -0.75,                                                   #8
    0.0, 0.0, -0.75,                                                    #9
    -0.25, -0.5, -1.0,                                                  #10
    0.0, 0.0, -1.0,                                                     #11
    0.0, 0.0, -0.5,                                                     #12
    -0.5, -1.0, -1.0,                                                   #13
    0.0, -1.0, 0.0,                                                     #14
    0.0, -1.5, -0.5                                                     #15
]

rot_max_list = [   
    glob_rot_range, glob_rot_range, glob_rot_range,                     #0
    0.25, 0.5, 1.5,                                                     #1
    0.0, 0.0, 1.5,                                                      #2
    0.0, 0.0, 1.5,                                                      #3
    0.25, 0.5, 1.5,                                                     #4
    0.0, 0.0, 1.5,                                                      #5
    0.0, 0.0, 1.5,                                                      #6
    0.25, 0.5, 1.5,                                                     #7
    1.0, 0.0, 1.5,                                                      #8
    0.0, 0.0, 1.5,                                                      #9
    0.25, 0.5, 1.5,                                                     #10
    0.0, 0.0, 1.5,                                                      #11
    0.0, 0.0, 1.5,                                                      #12
    -0.5, 0.5, 1.5,                                                     #13
    0.0, 0.5, 0.0,                                                      #14
    0.0, 1.0, -0.5                                                      #15
]

punish_cent_joint0 = 10000.0
punish_cent_joint1 = 10000.0
punish_cent_joint2 = 10000.0

punish_side_joint0 = 0.0
punish_side_joint1 = 10000.0
punish_side_joint2 = 10000.0

punish_vert_joint0 = 0.0
punish_vert_joint1 = 0.0
punish_vert_joint2 = 0.0

pose_reg_list = [
    # Finger: Index
    punish_cent_joint0, punish_side_joint0, punish_vert_joint0,     
    punish_cent_joint1, punish_side_joint1, punish_vert_joint1,            
    punish_cent_joint2, punish_side_joint2, punish_vert_joint2,   
    # Finger: Middle        
    punish_cent_joint0, punish_side_joint0, punish_vert_joint0,     
    punish_cent_joint1, punish_side_joint1, punish_vert_joint1,                     
    punish_cent_joint2, punish_side_joint2, punish_vert_joint2,   
    # Finger: Ring                
    punish_cent_joint0, punish_side_joint0, punish_vert_joint0,     
    punish_cent_joint1, punish_side_joint1, punish_vert_joint1,                     
    punish_cent_joint2, punish_side_joint2, punish_vert_joint2,    
    # Finger: Pinky            
    punish_cent_joint0, punish_side_joint0, punish_vert_joint0, 
    punish_cent_joint1, punish_side_joint1, punish_vert_joint1,                     
    punish_cent_joint2, punish_side_joint2, punish_vert_joint2,#38   
    # Finger: Thumb    
    punish_cent_joint0, punish_vert_joint0, punish_cent_joint0, 
    punish_cent_joint1, punish_vert_joint1, punish_side_joint1,
    punish_cent_joint2, punish_vert_joint2, punish_side_joint2                       
]

pose_mean_list = [
    0.012990133975529916, -0.2061533137671232, 0.1217358369522687, 
    0.0, 0.0, -0.26776195873424913, 
    0.0, 0.0, 0.1463492324860077, -0.019004827777825577, 
    0.045829039124687755, 0.008727056210913353, 0.0, 
    0.0, -0.1284752811149675, 
    0.0, 0.0, 0.23133631452056908, 
    -0.026223380872148847, 0.13406619518188134, -0.24682822100890117, 
    0.08579138535571655, 0.0, -0.08518110410666833, 
    0.0, 0.0, 0.5096552875298644, 
    -0.024652840864526005, -0.030070189038857012, -0.026405485406310773, 
    0.0, 0.0, -0.10829962287750862, 
    0.0, 0.0, 0.2650733518398053, 
    -0.5, -0.27061772045666727, 0.26407928733983543, 
    0.0, 0.07141815208289819, 0.0, 
    0.0, -0.23984496193594806, -0.5
]

rot0_idx = 0
rot1_idx = 1
rot2_idx = 2
fin0_ver_idx1 = 40
fin0_ver_idx2 = fin0_ver_idx1+3
fin0_ver_idx3 = fin0_ver_idx1+6
fin0_hor_idx1 = fin0_ver_idx1+1
fin0_rot_idx1 = fin0_ver_idx1-1

fin1_ver_idx1 = 5
fin1_ver_idx2 = fin1_ver_idx1+3
fin1_ver_idx3 = fin1_ver_idx1+6
fin1_hor_idx1 = fin1_ver_idx1-1
fin1_rot_idx1 = fin1_ver_idx1-2

fin2_ver_idx1 = 14
fin2_ver_idx2 = fin2_ver_idx1+3
fin2_ver_idx3 = fin2_ver_idx1+6
fin2_hor_idx1 = fin2_ver_idx1-1
fin2_rot_idx1 = fin2_ver_idx1-2

fin3_ver_idx1 = 32
fin3_ver_idx2 = fin3_ver_idx1+3
fin3_ver_idx3 = fin3_ver_idx1+6
fin3_hor_idx1 = fin3_ver_idx1-1
fin3_rot_idx1 = fin3_ver_idx1-2

fin4_ver_idx1 = 23
fin4_ver_idx2 = fin4_ver_idx1+3
fin4_ver_idx3 = fin4_ver_idx1+6
fin4_hor_idx1 = fin4_ver_idx1-1
fin4_rot_idx1 = fin4_ver_idx1-2
fin4_ver_fix_idx = 24

# For UI sliders
tick_count = 101
# xyz root
mano_xyz_root0_range =  np.linspace(-100, 100, tick_count)
mano_xyz_root1_range =  np.linspace(-100, 100, tick_count)
mano_xyz_root2_range =  np.linspace(0, 100, tick_count)
mano_xyz_root_range_list = [mano_xyz_root0_range, mano_xyz_root1_range, mano_xyz_root2_range]
# Shape
mano_shape_range = np.linspace(-10, 10, tick_count)
mano_shape_range_list = [mano_shape_range for i in range(10)]
# Global rotation
mano_glob_rot0_range = np.linspace(rot_min_list[rot0_idx], rot_max_list[rot0_idx], tick_count)
mano_glob_rot1_range = np.linspace(rot_min_list[rot1_idx], rot_max_list[rot1_idx], tick_count)
mano_glob_rot2_range = np.linspace(rot_min_list[rot2_idx], rot_max_list[rot2_idx], tick_count)
mano_glob_rot_range_list = [mano_glob_rot0_range, mano_glob_rot1_range, mano_glob_rot2_range]

# Finger 0 (thumb)
mano_fin0_ver_range1 = np.linspace(rot_min_list[fin0_ver_idx1], rot_max_list[fin0_ver_idx1], tick_count)
mano_fin0_ver_range2 = np.linspace(rot_min_list[fin0_ver_idx2], rot_max_list[fin0_ver_idx2], tick_count)
mano_fin0_ver_range3 = np.linspace(rot_min_list[fin0_ver_idx3], rot_max_list[fin0_ver_idx3], tick_count)
mano_fin0_hor_range1 = np.linspace(rot_min_list[fin0_hor_idx1], rot_max_list[fin0_hor_idx1], tick_count)
mano_fin0_rot_range1 = np.linspace(rot_min_list[fin0_rot_idx1], rot_max_list[fin0_rot_idx1], tick_count)
mano_finger0_rot_range_list = [mano_fin0_ver_range1, mano_fin0_ver_range2, mano_fin0_ver_range3, 
    mano_fin0_hor_range1, mano_fin0_rot_range1]
    
# Finger 1 (index)
mano_fin1_ver_range1 = np.linspace(rot_min_list[fin1_ver_idx1], rot_max_list[fin1_ver_idx1], tick_count)
mano_fin1_ver_range2 = np.linspace(rot_min_list[fin1_ver_idx2], rot_max_list[fin1_ver_idx2], tick_count)
mano_fin1_ver_range3 = np.linspace(rot_min_list[fin1_ver_idx3], rot_max_list[fin1_ver_idx3], tick_count)
mano_fin1_hor_range1 = np.linspace(rot_min_list[fin1_hor_idx1], rot_max_list[fin1_hor_idx1], tick_count)
mano_fin1_rot_range1 = np.linspace(rot_min_list[fin1_rot_idx1], rot_max_list[fin1_rot_idx1], tick_count)
mano_finger1_rot_range_list = [mano_fin1_ver_range1, mano_fin1_ver_range2, mano_fin1_ver_range3, 
    mano_fin1_hor_range1, mano_fin1_rot_range1]
    
# Finger 2 (middle)
mano_fin2_ver_range1 = np.linspace(rot_min_list[fin2_ver_idx1], rot_max_list[fin2_ver_idx1], tick_count)
mano_fin2_ver_range2 = np.linspace(rot_min_list[fin2_ver_idx2], rot_max_list[fin2_ver_idx2], tick_count)
mano_fin2_ver_range3 = np.linspace(rot_min_list[fin2_ver_idx3], rot_max_list[fin2_ver_idx3], tick_count)
mano_fin2_hor_range1 = np.linspace(rot_min_list[fin2_hor_idx1], rot_max_list[fin2_hor_idx1], tick_count)
mano_fin2_rot_range1 = np.linspace(rot_min_list[fin2_rot_idx1], rot_max_list[fin2_rot_idx1], tick_count)
mano_finger2_rot_range_list = [mano_fin2_ver_range1, mano_fin2_ver_range2, mano_fin2_ver_range3, 
    mano_fin2_hor_range1, mano_fin2_rot_range1]
    
# Finger 3 (ring)
mano_fin3_ver_range1 = np.linspace(rot_min_list[fin3_ver_idx1], rot_max_list[fin3_ver_idx1], tick_count)
mano_fin3_ver_range2 = np.linspace(rot_min_list[fin3_ver_idx2], rot_max_list[fin3_ver_idx2], tick_count)
mano_fin3_ver_range3 = np.linspace(rot_min_list[fin3_ver_idx3], rot_max_list[fin3_ver_idx3], tick_count)
mano_fin3_hor_range1 = np.linspace(rot_min_list[fin3_hor_idx1], rot_max_list[fin3_hor_idx1], tick_count)
mano_fin3_rot_range1 = np.linspace(rot_min_list[fin3_rot_idx1], rot_max_list[fin3_rot_idx1], tick_count)
mano_finger3_rot_range_list = [mano_fin3_ver_range1, mano_fin3_ver_range2, mano_fin3_ver_range3, 
    mano_fin3_hor_range1, mano_fin3_rot_range1]
    
# Finger 4 (pinky)
mano_fin4_ver_range1 = np.linspace(rot_min_list[fin4_ver_idx1], rot_max_list[fin4_ver_idx1], tick_count)
mano_fin4_ver_range2 = np.linspace(rot_min_list[fin4_ver_idx2], rot_max_list[fin4_ver_idx2], tick_count)
mano_fin4_ver_range3 = np.linspace(rot_min_list[fin4_ver_idx3], rot_max_list[fin4_ver_idx3], tick_count)
mano_fin4_hor_range1 = np.linspace(rot_min_list[fin4_hor_idx1], rot_max_list[fin4_hor_idx1], tick_count)
mano_fin4_rot_range1 = np.linspace(rot_min_list[fin4_rot_idx1], rot_max_list[fin4_rot_idx1], tick_count)
mano_finger4_rot_range_list = [mano_fin4_ver_range1, mano_fin4_ver_range2, mano_fin4_ver_range3, 
    mano_fin4_hor_range1, mano_fin4_rot_range1]
    
freeze_fin0_idx_range_tuple = (36, 45)
freeze_fin1_idx_range_tuple = (0, 9)
freeze_fin2_idx_range_tuple = (9, 18)
freeze_fin3_idx_range_tuple = (27, 36)
freeze_fin4_idx_range_tuple = (18, 27)
freeze_fin_idx_range_tuple_list = [freeze_fin0_idx_range_tuple, freeze_fin1_idx_range_tuple, freeze_fin2_idx_range_tuple, 
    freeze_fin3_idx_range_tuple, freeze_fin4_idx_range_tuple]