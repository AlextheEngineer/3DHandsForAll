import sys
import math
import cv2
import argparse
import numpy as np
import torch
from collections import OrderedDict
import tkinter as tk
from PIL import Image, ImageTk
import mano_wrapper
from utils import *
import params

class pose_annotation_app:
    def __init__(self, args):
        self.args = args
        self.init_window()
        self.samples_dir_path = 'sample_imgs'
        self.img_path_list = self.load_input_imgs()
        self.img_i = 0
        self.init_models()
        self.init_ui()
        
    def run_app(self):
        self.window.mainloop()
        
    def init_models(self):
        self.init_model_2d()
        self.init_model_3d_3rd()
        self.init_model_3d_ego()
        self.init_kpts_projector()
        
    def init_variables(self):
        self.mano_fit_tool = mano_wrapper.mano_fitter(params.MANO_PATH, is_annotating = True)
        self.mano_fit_tool.set_input_size(params.IMG_SIZE)
        self.img_left, self.img_center, self.img_right = None, None, None
        self.crop_center = None
        self.crop_size = params.CROP_SIZE_DEFAULT
        self.crop_box = None
        self.crop_info = None
        self.is_debugging = False
        self.img_side_is_left = None
        self.joint_selected = 0
        self.joint_anno_dict_l = dict()
        self.joint_anno_dict_r = dict()
        self.img_toggle_state = False
        self.mano_info_loaded = None
        self.kpt_2d_load_list = [i for i in range(21)]
        self.results_saved = False
        
    def load_input_imgs(self):
        # [Optional] Load your images here
        img_path_list = []
        img_path_list = [os.path.join(self.samples_dir_path, f_name) for f_name in os.listdir(self.samples_dir_path) if \
            f_name.endswith('.png') or f_name.endswith('.jpg') or f_name.endswith('.jpeg')]
        return img_path_list
        
    def init_window(self):
        self.window = tk.Tk()
        self.window.title('Hand Pose Annotator')
        window_height= self.window.winfo_screenheight() - 100
        window_width = params.IMG_SIZE*3 + params.IMG_PADDING*5
        self.window.geometry('{}x{}+0+0'.format(window_width, window_height))
        self.window.minsize(window_width, window_height)
        self.window.panel_top_ver = None
        
    def init_ui(self):
        print('Annotating {}'.format(self.img_path_list[self.img_i]))
        self.args.hand_mode = 'l'
        if self.window.panel_top_ver is not None:
            self.window.panel_top_ver.pack_forget()
            del self.window.panel_top_ver
        self.window.panel_top_ver = tk.PanedWindow(self.window, orient = tk.VERTICAL)
        self.window.panel_top_ver.pack(side = tk.TOP, anchor = tk.NW)
    
        self.window.panel_sliders = None
        self.window.panel_image_display = None
        self.window.panel_buttons = None
        
        self.init_variables()
        self.init_sliders()
        self.init_image_display()
        self.init_buttons()
        
        self.init_frame_info()
        self.update_slider_values()
        
    def init_frame_info(self):
        # [Optional] Load existing MANO file here
        mano_path = None
        # Init display images
        self.img_orig, self.img_left = load_img_mano(self.img_path_list[self.img_i])
        if self.args.hand_mode == 'l':
            self.img_left = cv2.flip(self.img_left, 1)
        self.img_right = np.zeros_like(self.img_left)
    
        # Init mano
        self.mano_info_loaded = None
        if mano_path is not None:
            self.mano_info_loaded = np.load(mano_path)
            root_idx = 9 if self.mano_mode_list[self.img_i] == 'old' else 0
            self.mano_fit_tool.set_mano(self.mano_info_loaded, root_idx = root_idx)
            self.img_center = self.get_rendered_img()
            # Init 2D kpt annotation
            kpts_2d_glob = np.rint(self.mano_fit_tool.get_kpts_2d_glob()).astype(np.int32)
            for kpt_load_i in self.kpt_2d_load_list:
                row_load, col_load = kpts_2d_glob[kpt_load_i]
                self.joint_anno_dict_l[kpt_load_i] = (row_load, col_load)
                self.mano_fit_tool.set_kpt_2d(kpt_load_i, np.array([row_load, col_load]))
            # Init crop box
            self.crop_center, self.crop_size = get_crop_attr_from_kpts_2d(kpts_2d_glob)
            self.update_crop_box()            
        else:    
            self.img_center = self.get_rendered_img()
        self.update_img_display()
        
    def init_sliders(self):
        if self.window.panel_sliders is not None:
            self.window.panel_sliders.pack_forget()
            del self.window.panel_sliders
    
        self.window.panel_sliders = tk.PanedWindow(self.window.panel_top_ver)
        self.window.panel_sliders.pack(fill = tk.X, expand=True, side=tk.TOP, anchor = tk.NW)
    
        num_rot_ticks = params.NUM_ROT_TICKS
        self.window.slider_joint = tk.Scale(self.window.panel_sliders, from_=0, to=20, label = 'joint idx', \
            command=self.on_trackbar_joint, orient=tk.HORIZONTAL)
        self.window.slider_joint.pack(fill = tk.X, padx = 10, side = tk.TOP)

        # Global rotation
        panel_hor_rot = tk.PanedWindow(self.window.panel_sliders, orient = tk.HORIZONTAL)
        panel_hor_rot.pack(fill = tk.X, expand=True, side = tk.TOP, anchor = tk.NW)

        slider_mano_rot0_name = 'rot0'
        self.window.slider_mano_rot0_value = tk.DoubleVar()
        slider_rot0 = tk.Scale(panel_hor_rot, variable = self.window.slider_mano_rot0_value, from_=0, to=num_rot_ticks, \
            label = slider_mano_rot0_name, command=self.on_trackbar_mano_rot0, orient=tk.HORIZONTAL)
        slider_rot0.pack(fill = tk.X, expand=True, padx = params.SCALE_PAD, side = tk.LEFT, anchor = tk.NW)

        slider_mano_rot1_name = 'rot1'
        self.window.slider_mano_rot1_value = tk.DoubleVar()
        slider_rot1 = tk.Scale(panel_hor_rot, variable = self.window.slider_mano_rot1_value, from_=0, to=num_rot_ticks, \
            label = slider_mano_rot1_name, command=self.on_trackbar_mano_rot1, orient=tk.HORIZONTAL)
        slider_rot1.pack(fill = tk.X, expand=True, padx = params.SCALE_PAD, side = tk.LEFT, anchor = tk.NW)

        slider_mano_rot2_name = 'rot2'
        self.window.slider_mano_rot2_value = tk.DoubleVar()
        slider_rot2 = tk.Scale(panel_hor_rot, variable = self.window.slider_mano_rot2_value, from_=0, to=num_rot_ticks, \
            label = slider_mano_rot2_name, command=self.on_trackbar_mano_rot2, orient=tk.HORIZONTAL) 
        slider_rot2.pack(fill = tk.X, expand=True, padx = params.SCALE_PAD, side = tk.LEFT, anchor = tk.NW)
        
        self.window.freeze_text_rot_variable = tk.StringVar()
        button_freeze_rot = tk.Button(panel_hor_rot, textvariable = self.window.freeze_text_rot_variable, bg = '#ffffff', \
            height = 1, width = 10, command = self.button_freeze_rot_callback)
        button_freeze_rot.pack(padx = params.SCALE_PAD, pady = params.SCALE_PAD, side = tk.LEFT, anchor = tk.W)
        self.window.freeze_text_rot_variable.set(params.FREEZE_TEXT)
        
        slider_mano_xyz_root0_name = 'x'
        self.window.slider_mano_xyz_root0_value = tk.DoubleVar()
        slider_xyz_root0 = tk.Scale(panel_hor_rot, variable = self.window.slider_mano_xyz_root0_value, from_=0, \
            to=num_rot_ticks, label = slider_mano_xyz_root0_name, command=self.on_trackbar_mano_xyz_root0, orient=tk.HORIZONTAL)
        slider_xyz_root0.pack(fill = tk.X, expand=True, padx = params.SCALE_PAD, side = tk.LEFT, anchor = tk.NW)
        self.window.slider_mano_xyz_root0_value.set(num_rot_ticks//2)

        slider_mano_xyz_root1_name = 'y'
        self.window.slider_mano_xyz_root1_value = tk.DoubleVar()
        slider_xyz_root1 = tk.Scale(panel_hor_rot, variable = self.window.slider_mano_xyz_root1_value, from_=0, \
            to=num_rot_ticks, label = slider_mano_xyz_root1_name, command=self.on_trackbar_mano_xyz_root1, orient=tk.HORIZONTAL)
        slider_xyz_root1.pack(fill = tk.X, expand=True, padx = params.SCALE_PAD, side = tk.LEFT, anchor = tk.NW)
        self.window.slider_mano_xyz_root1_value.set(num_rot_ticks//2)
        
        slider_mano_xyz_root2_name = 'z'
        self.window.slider_mano_xyz_root2_value = tk.DoubleVar()
        slider_xyz_root2 = tk.Scale(panel_hor_rot, variable = self.window.slider_mano_xyz_root2_value, from_=0, \
            to=num_rot_ticks, label = slider_mano_xyz_root2_name, command=self.on_trackbar_mano_xyz_root2, orient=tk.HORIZONTAL)
        slider_xyz_root2.pack(fill = tk.X, expand=True, padx = params.SCALE_PAD, side = tk.LEFT, anchor = tk.NW)
        self.window.slider_mano_xyz_root2_value.set(num_rot_ticks//2)

        # Thumb
        panel_hor_finger0 = tk.PanedWindow(self.window.panel_sliders, orient = tk.HORIZONTAL)
        panel_hor_finger0.pack(fill = tk.X, expand=True, side = tk.TOP, anchor = tk.NW)

        slider_mano_fin0_ver_name = 'Thumb |'
        self.window.slider_mano_fin0_ver_value = tk.DoubleVar()
        slider_finger0_rot0 = tk.Scale(panel_hor_finger0, variable = self.window.slider_mano_fin0_ver_value, from_=0, \
            to=num_rot_ticks, label = slider_mano_fin0_ver_name, command=self.on_trackbar_mano_finger0_ver, orient=tk.HORIZONTAL)
        slider_finger0_rot0.pack(fill = tk.X, expand=True, padx = params.SCALE_PAD, side = tk.LEFT, anchor = tk.NW)

        slider_mano_fin0_hor_name = 'Thumb -'
        self.window.slider_mano_fin0_hor_value = tk.DoubleVar()
        slider_finger0_rot1 = tk.Scale(panel_hor_finger0, variable = self.window.slider_mano_fin0_hor_value, from_=0, \
            to=num_rot_ticks, label = slider_mano_fin0_hor_name, command=self.on_trackbar_mano_finger0_hor, orient=tk.HORIZONTAL)
        slider_finger0_rot1.pack(fill = tk.X, expand=True, padx = params.SCALE_PAD, side = tk.LEFT, anchor = tk.NW)

        slider_mano_fin0_rot_name = 'Thumb o'
        self.window.slider_mano_fin0_rot_value = tk.DoubleVar()
        slider_finger0_rot2 = tk.Scale(panel_hor_finger0, variable = self.window.slider_mano_fin0_rot_value, from_=0, \
            to=num_rot_ticks, label = slider_mano_fin0_rot_name, command=self.on_trackbar_mano_finger0_rot, orient=tk.HORIZONTAL)
        slider_finger0_rot2.pack(fill = tk.X, expand=True, padx = params.SCALE_PAD, side = tk.LEFT, anchor = tk.NW)
        
        self.window.freeze_text_fin0_variable = tk.StringVar()
        button_freeze_fin0 = tk.Button(panel_hor_finger0, textvariable = self.window.freeze_text_fin0_variable, \
            bg = '#ffffff', height = 1, width = 10 , command = self.button_freeze_fin0_callback)
        button_freeze_fin0.pack(padx = params.SCALE_PAD, pady = params.SCALE_PAD, side = tk.LEFT, anchor = tk.W)
        self.window.freeze_text_fin0_variable.set(params.FREEZE_TEXT)
        
        slider_mano_fin0_ver0_name = 'Thumb |[0]'
        self.window.slider_mano_fin0_ver0_value = tk.DoubleVar()
        slider_finger0_0_rot0 = tk.Scale(panel_hor_finger0, variable = self.window.slider_mano_fin0_ver0_value, \
            from_=0, to=num_rot_ticks, label = slider_mano_fin0_ver0_name, command=self.on_trackbar_mano_finger0_ver0, orient=tk.HORIZONTAL)
        slider_finger0_0_rot0.pack(fill = tk.X, expand=True, padx = params.SCALE_PAD, side = tk.LEFT, anchor = tk.NW)
        
        slider_mano_fin0_ver1_name = 'Thumb |[1]'
        self.window.slider_mano_fin0_ver1_value = tk.DoubleVar()
        slider_finger0_1_rot0 = tk.Scale(panel_hor_finger0, variable = self.window.slider_mano_fin0_ver1_value, \
            from_=0, to=num_rot_ticks, label = slider_mano_fin0_ver1_name, command=self.on_trackbar_mano_finger0_ver1, orient=tk.HORIZONTAL)
        slider_finger0_1_rot0.pack(fill = tk.X, expand=True, padx = params.SCALE_PAD, side = tk.LEFT, anchor = tk.NW)
        
        slider_mano_fin0_ver2_name = 'Thumb |[2]'
        self.window.slider_mano_fin0_ver2_value = tk.DoubleVar()
        slider_finger0_2_rot0 = tk.Scale(panel_hor_finger0, variable = self.window.slider_mano_fin0_ver2_value, \
            from_=0, to=num_rot_ticks, label = slider_mano_fin0_ver2_name, command=self.on_trackbar_mano_finger0_ver2, orient=tk.HORIZONTAL)
        slider_finger0_2_rot0.pack(fill = tk.X, expand=True, padx = params.SCALE_PAD, side = tk.LEFT, anchor = tk.NW)

        # Index finger
        panel_hor_finger1 = tk.PanedWindow(self.window.panel_sliders, orient = tk.HORIZONTAL)
        panel_hor_finger1.pack(fill = tk.X, expand=True, side = tk.TOP, anchor = tk.NW)

        slider_mano_fin1_ver_name = 'Index |'
        self.window.slider_mano_fin1_ver_value = tk.DoubleVar()
        slider_finger1_rot0 = tk.Scale(panel_hor_finger1, variable = self.window.slider_mano_fin1_ver_value, from_=0, \
            to=num_rot_ticks, label = slider_mano_fin1_ver_name, command=self.on_trackbar_mano_finger1_ver, orient=tk.HORIZONTAL)
        slider_finger1_rot0.pack(fill = tk.X, expand=True, padx = params.SCALE_PAD, side = tk.LEFT, anchor = tk.NW)

        slider_mano_fin1_hor_name = 'Index -'
        self.window.slider_mano_fin1_hor_value = tk.DoubleVar()
        slider_finger1_rot1 = tk.Scale(panel_hor_finger1, variable = self.window.slider_mano_fin1_hor_value, from_=0, \
            to=num_rot_ticks, label = slider_mano_fin1_hor_name, command=self.on_trackbar_mano_finger1_hor, orient=tk.HORIZONTAL)
        slider_finger1_rot1.pack(fill = tk.X, expand=True, padx = params.SCALE_PAD, side = tk.LEFT, anchor = tk.NW)

        slider_mano_fin1_rot_name = 'Index o'
        self.window.slider_mano_fin1_rot_value = tk.DoubleVar()
        slider_finger1_rot2 = tk.Scale(panel_hor_finger1, variable = self.window.slider_mano_fin1_rot_value, from_=0, \
            to=num_rot_ticks, label = slider_mano_fin1_rot_name, command=self.on_trackbar_mano_finger1_rot, orient=tk.HORIZONTAL)
        slider_finger1_rot2.pack(fill = tk.X, expand=True, padx = params.SCALE_PAD, side = tk.LEFT, anchor = tk.NW)
        
        self.window.freeze_text_fin1_variable = tk.StringVar()
        button_freeze_fin1 = tk.Button(panel_hor_finger1, textvariable = self.window.freeze_text_fin1_variable, \
            bg = '#ffffff', height = 1, width = 10 , command = self.button_freeze_fin1_callback)
        button_freeze_fin1.pack(padx = params.SCALE_PAD, pady = params.SCALE_PAD, side = tk.LEFT, anchor = tk.W)
        self.window.freeze_text_fin1_variable.set(params.FREEZE_TEXT)
        
        slider_mano_fin1_ver0_name = 'Index |[0]'
        self.window.slider_mano_fin1_ver0_value = tk.DoubleVar()
        slider_finger1_0_rot0 = tk.Scale(panel_hor_finger1, variable = self.window.slider_mano_fin1_ver0_value, \
            from_=0, to=num_rot_ticks, label = slider_mano_fin1_ver0_name, command=self.on_trackbar_mano_finger1_ver0, orient=tk.HORIZONTAL)
        slider_finger1_0_rot0.pack(fill = tk.X, expand=True, padx = params.SCALE_PAD, side = tk.LEFT, anchor = tk.NW)
        
        slider_mano_fin1_ver1_name = 'Index |[1]'
        self.window.slider_mano_fin1_ver1_value = tk.DoubleVar()
        slider_finger1_1_rot0 = tk.Scale(panel_hor_finger1, variable = self.window.slider_mano_fin1_ver1_value, \
            from_=0, to=num_rot_ticks, label = slider_mano_fin1_ver1_name, command=self.on_trackbar_mano_finger1_ver1, orient=tk.HORIZONTAL)
        slider_finger1_1_rot0.pack(fill = tk.X, expand=True, padx = params.SCALE_PAD, side = tk.LEFT, anchor = tk.NW)
        
        slider_mano_fin1_ver2_name = 'Index |[2]'
        self.window.slider_mano_fin1_ver2_value = tk.DoubleVar()
        slider_finger1_2_rot0 = tk.Scale(panel_hor_finger1, variable = self.window.slider_mano_fin1_ver2_value, \
            from_=0, to=num_rot_ticks, label = slider_mano_fin1_ver2_name, command=self.on_trackbar_mano_finger1_ver2, orient=tk.HORIZONTAL)
        slider_finger1_2_rot0.pack(fill = tk.X, expand=True, padx = params.SCALE_PAD, side = tk.LEFT, anchor = tk.NW)

        # Middle finger
        panel_hor_finger2 = tk.PanedWindow(self.window.panel_sliders, orient = tk.HORIZONTAL)
        panel_hor_finger2.pack(fill = tk.X, expand=True, side = tk.TOP, anchor = tk.NW)

        slider_mano_fin2_ver_name = 'Middle |'
        self.window.slider_mano_fin2_ver_value = tk.DoubleVar()
        slider_finger2_rot0 = tk.Scale(panel_hor_finger2, variable = self.window.slider_mano_fin2_ver_value, \
            from_=0, to=num_rot_ticks, label = slider_mano_fin2_ver_name, command=self.on_trackbar_mano_finger2_ver, orient=tk.HORIZONTAL)
        slider_finger2_rot0.pack(fill = tk.X, expand=True, padx = params.SCALE_PAD, side = tk.LEFT, anchor = tk.NW)

        slider_mano_fin2_hor_name = 'Middle -'
        self.window.slider_mano_fin2_hor_value = tk.DoubleVar()
        slider_finger2_rot1 = tk.Scale(panel_hor_finger2, variable = self.window.slider_mano_fin2_hor_value, \
            from_=0, to=num_rot_ticks, label = slider_mano_fin2_hor_name, command=self.on_trackbar_mano_finger2_hor, orient=tk.HORIZONTAL)
        slider_finger2_rot1.pack(fill = tk.X, expand=True, padx = params.SCALE_PAD, side = tk.LEFT, anchor = tk.NW)

        slider_mano_fin2_rot_name = 'Middle o'
        self.window.slider_mano_fin2_rot_value = tk.DoubleVar()
        slider_finger2_rot2 = tk.Scale(panel_hor_finger2, variable = self.window.slider_mano_fin2_rot_value, \
            from_=0, to=num_rot_ticks, label = slider_mano_fin2_rot_name, command=self.on_trackbar_mano_finger2_rot, orient=tk.HORIZONTAL)
        slider_finger2_rot2.pack(fill = tk.X, expand=True, padx = params.SCALE_PAD, side = tk.LEFT, anchor = tk.NW)
        
        self.window.freeze_text_fin2_variable = tk.StringVar()
        button_freeze_fin2 = tk.Button(panel_hor_finger2, textvariable = self.window.freeze_text_fin2_variable, \
            bg = '#ffffff', height = 1, width = 10, command = self.button_freeze_fin2_callback)
        button_freeze_fin2.pack(padx = params.SCALE_PAD, pady = params.SCALE_PAD, side = tk.LEFT, anchor = tk.W)
        self.window.freeze_text_fin2_variable.set(params.FREEZE_TEXT)
        
        slider_mano_fin2_ver0_name = 'Middle |[0]'
        self.window.slider_mano_fin2_ver0_value = tk.DoubleVar()
        slider_finger2_0_rot0 = tk.Scale(panel_hor_finger2, variable = self.window.slider_mano_fin2_ver0_value, \
            from_=0, to=num_rot_ticks, label = slider_mano_fin2_ver0_name, command=self.on_trackbar_mano_finger2_ver0, orient=tk.HORIZONTAL)
        slider_finger2_0_rot0.pack(fill = tk.X, expand=True, padx = params.SCALE_PAD, side = tk.LEFT, anchor = tk.NW)
        
        slider_mano_fin2_ver1_name = 'Middle |[1]'
        self.window.slider_mano_fin2_ver1_value = tk.DoubleVar()
        slider_finger2_1_rot0 = tk.Scale(panel_hor_finger2, variable = self.window.slider_mano_fin2_ver1_value, \
            from_=0, to=num_rot_ticks, label = slider_mano_fin2_ver1_name, command=self.on_trackbar_mano_finger2_ver1, orient=tk.HORIZONTAL)
        slider_finger2_1_rot0.pack(fill = tk.X, expand=True, padx = params.SCALE_PAD, side = tk.LEFT, anchor = tk.NW)
        
        slider_mano_fin2_ver2_name = 'Middle |[2]'
        self.window.slider_mano_fin2_ver2_value = tk.DoubleVar()
        slider_finger2_2_rot0 = tk.Scale(panel_hor_finger2, variable = self.window.slider_mano_fin2_ver2_value, \
            from_=0, to=num_rot_ticks, label = slider_mano_fin2_ver2_name, command=self.on_trackbar_mano_finger2_ver2, orient=tk.HORIZONTAL)
        slider_finger2_2_rot0.pack(fill = tk.X, expand=True, padx = params.SCALE_PAD, side = tk.LEFT, anchor = tk.NW)

        # Ring finger
        panel_hor_finger3 = tk.PanedWindow(self.window.panel_sliders, orient = tk.HORIZONTAL) 
        panel_hor_finger3.pack(fill = tk.X, expand=True, side = tk.TOP, anchor = tk.NW)

        slider_mano_fin3_ver_name = 'Ring |'
        self.window.slider_mano_fin3_ver_value = tk.DoubleVar()
        slider_finger3_rot0 = tk.Scale(panel_hor_finger3, variable = self.window.slider_mano_fin3_ver_value, \
            from_=0, to=num_rot_ticks, label = slider_mano_fin3_ver_name, command=self.on_trackbar_mano_finger3_ver, orient=tk.HORIZONTAL)
        slider_finger3_rot0.pack(fill = tk.X, expand=True, padx = params.SCALE_PAD, side = tk.LEFT, anchor = tk.NW)

        slider_mano_fin3_hor_name = 'Ring -'
        self.window.slider_mano_fin3_hor_value = tk.DoubleVar()
        slider_finger3_rot1 = tk.Scale(panel_hor_finger3, variable = self.window.slider_mano_fin3_hor_value, \
            from_=0, to=num_rot_ticks, label = slider_mano_fin3_hor_name, command=self.on_trackbar_mano_finger3_hor, orient=tk.HORIZONTAL)
        slider_finger3_rot1.pack(fill = tk.X, expand=True, padx = params.SCALE_PAD, side = tk.LEFT, anchor = tk.NW)

        slider_mano_fin3_rot_name = 'Ring o'
        self.window.slider_mano_fin3_rot_value = tk.DoubleVar()
        slider_finger3_rot2 = tk.Scale(panel_hor_finger3, variable = self.window.slider_mano_fin3_rot_value, \
            from_=0, to=num_rot_ticks, label = slider_mano_fin3_rot_name, command=self.on_trackbar_mano_finger3_rot, orient=tk.HORIZONTAL)
        slider_finger3_rot2.pack(fill = tk.X, expand=True, padx = params.SCALE_PAD, side = tk.LEFT, anchor = tk.NW)
        
        self.window.freeze_text_fin3_variable = tk.StringVar()
        button_freeze_fin3 = tk.Button(panel_hor_finger3, textvariable = self.window.freeze_text_fin3_variable, \
            bg = '#ffffff', height = 1, width = 10, command = self.button_freeze_fin3_callback)
        button_freeze_fin3.pack(padx = params.SCALE_PAD, pady = params.SCALE_PAD, side = tk.LEFT, anchor = tk.W)
        self.window.freeze_text_fin3_variable.set(params.FREEZE_TEXT)
        
        slider_mano_fin3_ver0_name = 'Ring |[0]'
        self.window.slider_mano_fin3_ver0_value = tk.DoubleVar()
        slider_finger3_0_rot0 = tk.Scale(panel_hor_finger3, variable = self.window.slider_mano_fin3_ver0_value, \
            from_=0, to=num_rot_ticks, label = slider_mano_fin3_ver0_name, command=self.on_trackbar_mano_finger3_ver0, orient=tk.HORIZONTAL)
        slider_finger3_0_rot0.pack(fill = tk.X, expand=True, padx = params.SCALE_PAD, side = tk.LEFT, anchor = tk.NW)
        
        slider_mano_fin3_ver1_name = 'Ring |[1]'
        self.window.slider_mano_fin3_ver1_value = tk.DoubleVar()
        slider_finger3_1_rot0 = tk.Scale(panel_hor_finger3, variable = self.window.slider_mano_fin3_ver1_value, \
            from_=0, to=num_rot_ticks, label = slider_mano_fin3_ver1_name, command=self.on_trackbar_mano_finger3_ver1, orient=tk.HORIZONTAL)
        slider_finger3_1_rot0.pack(fill = tk.X, expand=True, padx = params.SCALE_PAD, side = tk.LEFT, anchor = tk.NW)
        
        slider_mano_fin3_ver2_name = 'Ring |[2]'
        self.window.slider_mano_fin3_ver2_value = tk.DoubleVar()
        slider_finger3_2_rot0 = tk.Scale(panel_hor_finger3, variable = self.window.slider_mano_fin3_ver2_value, \
            from_=0, to=num_rot_ticks, label = slider_mano_fin3_ver2_name, command=self.on_trackbar_mano_finger3_ver2, orient=tk.HORIZONTAL)
        slider_finger3_2_rot0.pack(fill = tk.X, expand=True, padx = params.SCALE_PAD, side = tk.LEFT, anchor = tk.NW)

        # Pinky finger
        panel_hor_finger4 = tk.PanedWindow(self.window.panel_sliders, orient = tk.HORIZONTAL)
        panel_hor_finger4.pack(fill = tk.X, expand=True, side = tk.TOP, anchor = tk.NW)

        slider_mano_fin4_ver_name = 'Pinky |'
        self.window.slider_mano_fin4_ver_value = tk.DoubleVar()
        slider_finger4_rot0 = tk.Scale(panel_hor_finger4, variable = self.window.slider_mano_fin4_ver_value, \
            from_=0, to=num_rot_ticks, label = slider_mano_fin4_ver_name, command=self.on_trackbar_mano_finger4_ver, orient=tk.HORIZONTAL)
        slider_finger4_rot0.pack(fill = tk.X, expand=True, padx = params.SCALE_PAD, side = tk.LEFT, anchor = tk.NW)

        slider_mano_fin4_hor_name = 'Pinky -'
        self.window.slider_mano_fin4_hor_value = tk.DoubleVar()
        slider_finger4_rot1 = tk.Scale(panel_hor_finger4, variable = self.window.slider_mano_fin4_hor_value, \
            from_=0, to=num_rot_ticks, label = slider_mano_fin4_hor_name, command=self.on_trackbar_mano_finger4_hor, orient=tk.HORIZONTAL)
        slider_finger4_rot1.pack(fill = tk.X, expand=True, padx = params.SCALE_PAD, side = tk.LEFT, anchor = tk.NW)

        slider_mano_fin4_rot_name = 'Pinky o'
        self.window.slider_mano_fin4_rot_value = tk.DoubleVar()
        slider_finger4_rot2 = tk.Scale(panel_hor_finger4, variable = self.window.slider_mano_fin4_rot_value, \
            from_=0, to=num_rot_ticks, label = slider_mano_fin4_rot_name, command=self.on_trackbar_mano_finger4_rot, orient=tk.HORIZONTAL)
        slider_finger4_rot2.pack(fill = tk.X, expand=True, padx = params.SCALE_PAD, side = tk.LEFT, anchor = tk.NW)
        
        self.window.freeze_text_fin4_variable = tk.StringVar()
        button_freeze_fin4 = tk.Button(panel_hor_finger4, textvariable = self.window.freeze_text_fin4_variable, \
            bg = '#ffffff', height = 1, width = 10, command = self.button_freeze_fin4_callback)
        button_freeze_fin4.pack(padx = params.SCALE_PAD, pady = params.SCALE_PAD, side = tk.LEFT, anchor = tk.W)
        self.window.freeze_text_fin4_variable.set(params.FREEZE_TEXT)
        
        slider_mano_fin4_ver0_name = 'Pinky |[0]'
        self.window.slider_mano_fin4_ver0_value = tk.DoubleVar()
        slider_finger4_0_rot0 = tk.Scale(panel_hor_finger4, variable = self.window.slider_mano_fin4_ver0_value, \
            from_=0, to=num_rot_ticks, label = slider_mano_fin4_ver0_name, command=self.on_trackbar_mano_finger4_ver0, orient=tk.HORIZONTAL)
        slider_finger4_0_rot0.pack(fill = tk.X, expand=True, padx = params.SCALE_PAD, side = tk.LEFT, anchor = tk.NW)
        
        slider_mano_fin4_ver1_name = 'Pinky |[1]'
        self.window.slider_mano_fin4_ver1_value = tk.DoubleVar()
        slider_finger4_1_rot0 = tk.Scale(panel_hor_finger4, variable = self.window.slider_mano_fin4_ver1_value, \
            from_=0, to=num_rot_ticks, label = slider_mano_fin4_ver1_name, command=self.on_trackbar_mano_finger4_ver1, orient=tk.HORIZONTAL)
        slider_finger4_1_rot0.pack(fill = tk.X, expand=True, padx = params.SCALE_PAD, side = tk.LEFT, anchor = tk.NW)
        
        slider_mano_fin4_ver2_name = 'Pinky |[2]'
        self.window.slider_mano_fin4_ver2_value = tk.DoubleVar()
        slider_finger4_2_rot0 = tk.Scale(panel_hor_finger4, variable = self.window.slider_mano_fin4_ver2_value, \
            from_=0, to=num_rot_ticks, label = slider_mano_fin4_ver2_name, command=self.on_trackbar_mano_finger4_ver2, orient=tk.HORIZONTAL)
        slider_finger4_2_rot0.pack(fill = tk.X, expand=True, padx = params.SCALE_PAD, side = tk.LEFT, anchor = tk.NW)
        
    def init_image_display(self):
        self.img_display_np = np.ones((params.IMG_SIZE+params.IMG_PADDING*2, \
            (params.IMG_SIZE+params.IMG_PADDING*2)*3, 3), dtype=np.uint8)*128
        if self.window.panel_image_display is not None:
            self.window.panel_image_display.pack_forget()
            del self.window.panel_image_display
        self.window.panel_image_display = tk.PanedWindow(self.window.panel_top_ver)
        self.window.panel_image_display.pack(pady = params.SCALE_PAD)
        self.window.image_display = ImageTk.PhotoImage(Image.fromarray(np.zeros_like(self.img_display_np)))
        self.window.canvas = tk.Canvas(self.window.panel_image_display, width=params.IMG_SIZE*3 + params.IMG_PADDING*6, \
            height=params.IMG_SIZE + params.IMG_PADDING*2)
        self.window.canvas_img = self.window.canvas.create_image(0,0,image=self.window.image_display,anchor='nw')
        self.window.canvas.config(scrollregion=self.window.canvas.bbox(tk.ALL))
        self.window.canvas.bind('<Button 1>', self.mouse_left_click_callback)
        self.window.canvas.bind('<Button 3>', self.mouse_right_click_callback)
        self.window.canvas.bind('<MouseWheel>', self.mouse_wheel_callback)
        self.window.canvas.pack(fill=tk.X,)
    
    def init_buttons(self):
        button_x_padding = params.BUTTON_X_PAD
        button_h, button_w = params.BUTTON_H, params.BUTTON_W
        scale_pad = params.SCALE_PAD
        
        if self.window.panel_buttons is not None:
            self.window.panel_buttons.pack_forget()
            del self.window.panel_buttons
        self.window.panel_buttons = tk.PanedWindow(self.window.panel_top_ver, orient = tk.HORIZONTAL)
        self.window.panel_buttons.pack(padx = scale_pad, pady = 20)

        button_prev = tk.Button(self.window.panel_buttons, text ='Prev', bg = '#ffffff', \
            height = button_h, width = button_w, command = self.button_prev_callback)
        button_prev.pack(padx = button_x_padding, pady = scale_pad, side = tk.LEFT)

        button_save = tk.Button(self.window.panel_buttons, text ='Save', bg = '#00d68f', \
            height = button_h, width = button_w, command = self.button_save_callback)
        button_save.pack(padx = button_x_padding, pady = scale_pad, side = tk.RIGHT)

        button_next = tk.Button(self.window.panel_buttons, text ='Next', bg = '#ffffff', \
            height = button_h, width = button_w, command = self.button_next_callback)
        button_next.pack(padx = button_x_padding, pady = scale_pad, side = tk.RIGHT)

        panel_op_buttons = tk.PanedWindow(self.window.panel_buttons, orient = tk.VERTICAL)
        panel_op_buttons.pack(padx = scale_pad, pady = scale_pad)
        
        panel_op_layer1_buttons = tk.PanedWindow(panel_op_buttons, orient = tk.HORIZONTAL)
        panel_op_layer1_buttons.pack(padx = scale_pad, pady = scale_pad)

        button_flip = tk.Button(panel_op_layer1_buttons, text ='Flip', bg = '#eba834', \
            height = button_h//3, width = button_w, command = self.button_flip_callback)
        button_flip.pack(padx = scale_pad, pady = scale_pad, side = tk.LEFT)

        button_toggle = tk.Button(panel_op_layer1_buttons, text ='Toggle', bg = '#e7dab6', \
            height = button_h//3, width = button_w, command = self.button_toggle_callback)
        button_toggle.pack(padx = scale_pad, pady = scale_pad, side = tk.LEFT)
        
        button_pred_2d = tk.Button(panel_op_layer1_buttons, text ='Pred 2D', bg = '#4eded9', \
            height = button_h//3, width = button_w, command = self.button_pred_2d_callback)
        button_pred_2d.pack(padx = scale_pad, pady = scale_pad, side = tk.LEFT)
        
        panel_op_layer2_buttons = tk.PanedWindow(panel_op_buttons, orient = tk.HORIZONTAL)
        panel_op_layer2_buttons.pack(padx = scale_pad, pady = scale_pad)
        
        button_fit_3d_3rd = tk.Button(panel_op_layer2_buttons, text ='Pred 3D 3rd', bg = '#dbc84b', \
            height = button_h//3, width = button_w, command = self.button_fit_3d_3rd_callback)
        button_fit_3d_3rd.pack(padx = scale_pad, pady = scale_pad, side = tk.LEFT)
        
        button_pred_3d_ego = tk.Button(panel_op_layer2_buttons, text ='Pred 3D ego', bg = '#e0a82f', \
            height = button_h//3, width = button_w, command = self.button_fit_3d_ego_callback)
        button_pred_3d_ego.pack(padx = scale_pad, pady = scale_pad, side = tk.LEFT)
        
        button_fit_root = tk.Button(panel_op_layer2_buttons, text ='Fit root', bg = '#42dcff', \
            height = button_h//3, width = button_w, command = self.button_fit_root_callback)
        button_fit_root.pack(padx = scale_pad, pady = scale_pad, side = tk.LEFT)
        
        button_fit_2d = tk.Button(panel_op_layer2_buttons, text ='Fit 2D', bg = '#D100FF', \
            height = button_h//3, width = button_w, command = self.button_fit_2d_callback)
        button_fit_2d.pack(padx = scale_pad, pady = scale_pad, side = tk.LEFT)
        
        panel_op_layer3_buttons = tk.PanedWindow(panel_op_buttons, orient = tk.HORIZONTAL)
        panel_op_layer3_buttons.pack(padx = scale_pad, pady = scale_pad)
        
        button_reset = tk.Button(panel_op_layer3_buttons, text ='Reset', bg = '#fc2847', \
            height = button_h//3, width = button_w, command = self.button_reset_callback)
        button_reset.pack(padx = scale_pad, pady = scale_pad, side = tk.LEFT)
        
    def init_model_2d(self):
        from models.HRNet.config import cfg
        from models.HRNet.config import update_config
        from models.HRNet.models import pose_hrnet
        update_config(cfg)
        model = pose_hrnet.get_pose_net(cfg, is_train=False)
        pretrained_2d_path = params.MODEL_2D_PATH
        if not os.path.exists(pretrained_2d_path):
            self.model_2d = None
            print('Model 2D not found : {}'.format(pretrained_2d_path))
        else:
            print('Loading {}'.format(pretrained_2d_path))
            print('model_2d #params: {}'.format(sum(p.numel() for p in model.parameters())))
            state_dict = torch.load(pretrained_2d_path)['state_dict']
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
            self.model_2d = torch.nn.DataParallel(model, device_ids=[0]).cuda()
            self.model_2d.eval()
            print('Model 2D succesfully loaded')
            
    def init_model_3d_3rd(self):
        from models.resnet import resnet_pose
        model = resnet_pose.resnet9(num_classes = params.NUM_KPTS*3, num_inchan=params.NUM_KPTS)
        pretrained_3d_path = params.MODEL_3D_3RD_PATH
        if not os.path.exists(pretrained_3d_path):
            self.model_3d_3rd = None
            print('Model third-person 3D not found : {}'.format(pretrained_3d_path))
        else:
            print('Loading {}'.format(pretrained_3d_path))
            print('model_3d third-person #params: {}'.format(sum(p.numel() for p in model.parameters())))
            state_dict = torch.load(pretrained_3d_path)['state_dict']
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
            self.model_3d_3rd = torch.nn.DataParallel(model, device_ids=[0]).cuda()
            self.model_3d_3rd.eval()
            print('Model third-person 3D succesfully loaded')
            
    def init_model_3d_ego(self):
        from models.resnet import resnet_pose
        model = resnet_pose.resnet9(num_classes = params.NUM_KPTS*3, num_inchan=params.NUM_KPTS)
        pretrained_3d_path = params.MODEL_3D_EGO_PATH
        if not os.path.exists(pretrained_3d_path):
            self.model_3d_ego = None
            print('Model egocentric 3D not found : {}'.format(pretrained_3d_path))
        else:
            print('Loading {}'.format(pretrained_3d_path))
            print('model_3d egocentric #params: {}'.format(sum(p.numel() for p in model.parameters())))
            state_dict = torch.load(pretrained_3d_path)['state_dict']
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
            self.model_3d_ego = torch.nn.DataParallel(model, device_ids=[0]).cuda()
            self.model_3d_ego.eval()
            print('Model egocentric 3D succesfully loaded')
            
    def init_kpts_projector(self):
        from kpts_global_projector import kpts_global_projector
        self.kpts_global_project_tool = kpts_global_projector('mano_hand')
        
    # Utility functions
    def get_rendered_img(self):
        return self.mano_fit_tool.get_rendered_img()
    
    def in_img_left_range(self, row, col):
        if (row >= params.IMG_PADDING and row < params.IMG_SIZE + params.IMG_PADDING) and \
            (col >= params.IMG_PADDING and col < params.IMG_SIZE + params.IMG_PADDING):
            return True
        else:
            return False
            
    def in_img_right_range(self, row, col):
        if (row >= params.IMG_PADDING and row < params.IMG_SIZE + params.IMG_PADDING) and \
            (col >= params.IMG_PADDING*5+params.IMG_SIZE*2 and col < params.IMG_PADDING*5+params.IMG_SIZE*3):
            return True
        else:
            return False
            
    def toggle_freeze_restore_text(self, tk_text_var, current_freeze_state):
        if current_freeze_state:
            tk_text_var.set(params.RESTORE_TEXT)
        else:
            tk_text_var.set(params.FREEZE_TEXT)
            # Restore global rotation
            self.update_rendered_img()
                
    def toggle_freeze_finger_text(self, tk_text_var, current_freeze_state):
        if current_freeze_state:
            tk_text_var.set(params.RESTORE_TEXT)
        else:
            tk_text_var.set(params.FREEZE_TEXT)
            # Restore global rotation
            self.update_rendered_img()
            
    def generate_joint_anno_dict_r(self):
        assert self.crop_info is not None
        for joint_idx in self.joint_anno_dict_l:
            row_orig, col_orig = self.joint_anno_dict_l[joint_idx]
            self.synch_kpt_2d_annotation('left', joint_idx, row_orig, col_orig)
            
    def get_kpts_2d_glob_gt_from_dict(self):
        kpts_2d_glob_np = np.zeros((params.NUM_KPTS, 2))
        if len(self.joint_anno_dict_l) != params.NUM_KPTS:
            print("Error, incorrect #target 2D kpt of {}".format(len(self.joint_anno_dict_l)))
            return
        for joint_idx in range(params.NUM_KPTS):
            row, col = self.joint_anno_dict_l[joint_idx]
            kpts_2d_glob_np[joint_idx] = np.array([row, col])
        return kpts_2d_glob_np
            
    # Reset functions
    def reset_crop_box(self):
        self.crop_center = None
        self.crop_box = None
        self.crop_info = None
        self.img_right = np.zeros_like(self.img_left)
        self.joint_anno_dict_r = dict()
            
    # Update functions
    def update_img_display(self):
        # Display left image
        row_s_l = params.IMG_PADDING
        row_e_l = row_s_l + params.IMG_SIZE
        col_s_l = params.IMG_PADDING
        col_e_l = col_s_l + params.IMG_SIZE
        
        img_left_display = self.img_left.copy()
        # Display annotated 2D joints
        for joint_idx in self.joint_anno_dict_l:
            row_click, col_click = self.joint_anno_dict_l[joint_idx]
            img_left_display = cv2.circle(img_left_display, (col_click, row_click), 3, (255, 0, 0), -1)
            img_left_display = cv2.putText(img_left_display, ' {}'.format(str(joint_idx)), \
                (col_click, row_click), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        # Display cropping box
        if self.crop_box is not None:
            img_left_display = cv2.rectangle(img_left_display, (self.crop_box[2], self.crop_box[0]), \
                (self.crop_box[3], self.crop_box[1]), (0,255,0), 1)
            
        self.img_display_np[row_s_l:row_e_l, col_s_l:col_e_l, :] = img_left_display
        
        # Display center image
        row_s_c = row_s_l
        row_e_c = row_e_l
        col_s_c = col_e_l + params.IMG_PADDING*2
        col_e_c = col_s_c + params.IMG_SIZE
        
        self.img_display_np[row_s_c:row_e_c, col_s_c:col_e_c, :] = self.img_center
        
        # Display right image
        row_s_r = row_s_l
        row_e_r = row_e_l
        col_s_r = col_e_c + params.IMG_PADDING*2
        col_e_r = col_s_r + params.IMG_SIZE
        
        img_right_display = self.img_right.copy()
        # Display annotated 2D joints
        for joint_idx in self.joint_anno_dict_r:
            row_click, col_click = self.joint_anno_dict_r[joint_idx]
            img_right_display = cv2.circle(img_right_display, (col_click, row_click), 5, (255, 0, 0), -1)
            img_right_display = cv2.putText(img_right_display, ' {}'.format(str(joint_idx)), \
                (col_click, row_click), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        
        self.img_display_np[row_s_r:row_e_r, col_s_r:col_e_r, :] = img_right_display
        
        self.window.image_display = ImageTk.PhotoImage(Image.fromarray(self.img_display_np))
        self.window.canvas.itemconfig(self.window.canvas_img, image=self.window.image_display)
    
    def update_slider_values(self):
        self.window.slider_mano_xyz_root0_value.set(get_slider_from_mano_val('xyz_root', 0, \
            self.mano_fit_tool.get_xyz_root_i(0)))
        self.window.slider_mano_xyz_root1_value.set(get_slider_from_mano_val('xyz_root', 1, \
            self.mano_fit_tool.get_xyz_root_i(1)))
        self.window.slider_mano_xyz_root2_value.set(get_slider_from_mano_val('xyz_root', 2, \
            self.mano_fit_tool.get_xyz_root_i(2)))
        self.update_slider_rot()
        self.update_slider_values_fin0()
        self.update_slider_values_fin1()
        self.update_slider_values_fin2()
        self.update_slider_values_fin3()
        self.update_slider_values_fin4()

    def update_slider_rot(self):
        self.window.slider_mano_rot0_value.set(get_slider_from_mano_val('glob', 0, self.mano_fit_tool.get_input_rot_i(0)))
        self.window.slider_mano_rot1_value.set(get_slider_from_mano_val('glob', 1, self.mano_fit_tool.get_input_rot_i(1)))
        self.window.slider_mano_rot2_value.set(get_slider_from_mano_val('glob', 2, self.mano_fit_tool.get_input_rot_i(2)))

    def update_slider_values_fin0(self):
        self.window.slider_mano_fin0_ver_value.set(get_slider_from_mano_val('finger0', 0, \
            self.mano_fit_tool.get_input_pose_i(params.fin0_ver_idx1 - 3)))
        self.window.slider_mano_fin0_ver0_value.set(get_slider_from_mano_val('finger0', 0, \
            self.mano_fit_tool.get_input_pose_i(params.fin0_ver_idx1 - 3)))
        self.window.slider_mano_fin0_ver1_value.set(get_slider_from_mano_val('finger0', 1, \
            self.mano_fit_tool.get_input_pose_i(params.fin0_ver_idx2 - 3)))
        self.window.slider_mano_fin0_ver2_value.set(get_slider_from_mano_val('finger0', 2, \
            self.mano_fit_tool.get_input_pose_i(params.fin0_ver_idx3 - 3)))
        self.window.slider_mano_fin0_hor_value.set(get_slider_from_mano_val('finger0', 3, \
            self.mano_fit_tool.get_input_pose_i(params.fin0_hor_idx1 - 3)))
        self.window.slider_mano_fin0_rot_value.set(get_slider_from_mano_val('finger0', 4, \
            self.mano_fit_tool.get_input_pose_i(params.fin0_rot_idx1 - 3)))
    
    def update_slider_values_fin1(self):
        self.window.slider_mano_fin1_ver_value.set(get_slider_from_mano_val('finger1', 0, \
            self.mano_fit_tool.get_input_pose_i(params.fin1_ver_idx1 - 3)))
        self.window.slider_mano_fin1_ver0_value.set(get_slider_from_mano_val('finger1', 0, \
            self.mano_fit_tool.get_input_pose_i(params.fin1_ver_idx1 - 3)))
        self.window.slider_mano_fin1_ver1_value.set(get_slider_from_mano_val('finger1', 1, \
            self.mano_fit_tool.get_input_pose_i(params.fin1_ver_idx2 - 3)))
        self.window.slider_mano_fin1_ver2_value.set(get_slider_from_mano_val('finger1', 2, \
            self.mano_fit_tool.get_input_pose_i(params.fin1_ver_idx3 - 3)))
        self.window.slider_mano_fin1_hor_value.set(get_slider_from_mano_val('finger1', 3, \
            self.mano_fit_tool.get_input_pose_i(params.fin1_hor_idx1 - 3)))
        self.window.slider_mano_fin1_rot_value.set(get_slider_from_mano_val('finger1', 4, \
            self.mano_fit_tool.get_input_pose_i(params.fin1_rot_idx1 - 3)))
        
    def update_slider_values_fin2(self):
        self.window.slider_mano_fin2_ver_value.set(get_slider_from_mano_val('finger2', 0, \
            self.mano_fit_tool.get_input_pose_i(params.fin2_ver_idx1 - 3)))
        self.window.slider_mano_fin2_ver0_value.set(get_slider_from_mano_val('finger2', 0, \
            self.mano_fit_tool.get_input_pose_i(params.fin2_ver_idx1 - 3)))
        self.window.slider_mano_fin2_ver1_value.set(get_slider_from_mano_val('finger2', 1, \
            self.mano_fit_tool.get_input_pose_i(params.fin2_ver_idx2 - 3)))
        self.window.slider_mano_fin2_ver2_value.set(get_slider_from_mano_val('finger2', 2, \
            self.mano_fit_tool.get_input_pose_i(params.fin2_ver_idx3 - 3)))
        self.window.slider_mano_fin2_hor_value.set(get_slider_from_mano_val('finger2', 3, \
            self.mano_fit_tool.get_input_pose_i(params.fin2_hor_idx1 - 3)))
        self.window.slider_mano_fin2_rot_value.set(get_slider_from_mano_val('finger2', 4, \
            self.mano_fit_tool.get_input_pose_i(params.fin2_rot_idx1 - 3)))
        
    def update_slider_values_fin3(self):
        self.window.slider_mano_fin3_ver_value.set(get_slider_from_mano_val('finger3', 0, \
            self.mano_fit_tool.get_input_pose_i(params.fin3_ver_idx1 - 3)))
        self.window.slider_mano_fin3_ver0_value.set(get_slider_from_mano_val('finger3', 0, \
            self.mano_fit_tool.get_input_pose_i(params.fin3_ver_idx1 - 3)))
        self.window.slider_mano_fin3_ver1_value.set(get_slider_from_mano_val('finger3', 1, \
            self.mano_fit_tool.get_input_pose_i(params.fin3_ver_idx2 - 3)))
        self.window.slider_mano_fin3_ver2_value.set(get_slider_from_mano_val('finger3', 2, \
            self.mano_fit_tool.get_input_pose_i(params.fin3_ver_idx3 - 3)))
        self.window.slider_mano_fin3_hor_value.set(get_slider_from_mano_val('finger3', 3, \
            self.mano_fit_tool.get_input_pose_i(params.fin3_hor_idx1 - 3)))
        self.window.slider_mano_fin3_rot_value.set(get_slider_from_mano_val('finger3', 4, \
            self.mano_fit_tool.get_input_pose_i(params.fin3_rot_idx1 - 3)))
            
    def update_slider_values_fin4(self):
        self.window.slider_mano_fin4_ver_value.set(get_slider_from_mano_val('finger4', 0, \
            self.mano_fit_tool.get_input_pose_i(params.fin4_ver_idx1 - 3)))
        self.window.slider_mano_fin4_ver0_value.set(get_slider_from_mano_val('finger4', 0, \
            self.mano_fit_tool.get_input_pose_i(params.fin4_ver_idx1 - 3)))
        self.window.slider_mano_fin4_ver1_value.set(get_slider_from_mano_val('finger4', 1, \
            self.mano_fit_tool.get_input_pose_i(params.fin4_ver_idx2 - 3)))
        self.window.slider_mano_fin4_ver2_value.set(get_slider_from_mano_val('finger4', 2, \
            self.mano_fit_tool.get_input_pose_i(params.fin4_ver_idx3 - 3)))
        self.window.slider_mano_fin4_hor_value.set(get_slider_from_mano_val('finger4', 3, \
            self.mano_fit_tool.get_input_pose_i(params.fin4_hor_idx1 - 3)))
        self.window.slider_mano_fin4_rot_value.set(get_slider_from_mano_val('finger4', 4, \
            self.mano_fit_tool.get_input_pose_i(params.fin4_rot_idx1 - 3)))
            
    def update_crop_box(self):
        # Set crop box
        crop_top = round(self.crop_center[0] - self.crop_size//2)
        crop_bot = crop_top + self.crop_size
        crop_left = round(self.crop_center[1] - self.crop_size//2)
        crop_right = crop_left + self.crop_size
        self.crop_box = (crop_top, crop_bot, crop_left, crop_right)
        # Update img_right
        padding_top = max(0, -crop_top)
        padding_bot = max(0, crop_bot - params.IMG_SIZE)
        padding_left = max(0, -crop_left)
        padding_right = max(0, crop_right - params.IMG_SIZE)
        img_crop_top = max(0, crop_top)
        img_crop_bot = min(params.IMG_SIZE, crop_bot)
        img_crop_left = max(0, crop_left)
        img_crop_right = min(params.IMG_SIZE, crop_right)
        img_cropped = self.img_left[img_crop_top:img_crop_bot, img_crop_left:img_crop_right]
        img_padded = cv2.copyMakeBorder(img_cropped, top=padding_top, bottom=padding_bot, \
            left=padding_left, right=padding_right, borderType=cv2.BORDER_CONSTANT)
        crop_h = (img_crop_bot - img_crop_top) + (padding_top + padding_bot)
        crop_w = (img_crop_right - img_crop_left) + (padding_left + padding_right)
        self.crop_info = (img_crop_top, img_crop_bot, img_crop_left, img_crop_right, \
            padding_top, padding_bot, padding_left, padding_right, crop_h, crop_w)
        assert_valid_crop(self.crop_info)
        self.img_right = cv2.resize(img_padded, (params.IMG_SIZE, params.IMG_SIZE))
        # Update 2D kpts on cropped image
        self.generate_joint_anno_dict_r()
        self.update_img_display()
        
    def add_kpt_2d_annotation_left(self, synch_mode, joint_idx, row, col):
        row, col = int(row), int(col)
        self.joint_anno_dict_l[joint_idx] = (row, col)
        if not synch_mode:
            self.synch_kpt_2d_annotation('left', joint_idx, row, col)
        self.mano_fit_tool.set_kpt_2d(joint_idx, np.array([row, col]))
        self.update_img_display()
        
    def add_kpt_2d_annotation_right(self, synch_mode, joint_idx, row, col):
        row, col = int(row), int(col)
        self.joint_anno_dict_r[joint_idx] = (row, col)
        if not synch_mode:
            self.synch_kpt_2d_annotation('right', joint_idx, row, col)
        self.update_img_display()
        
    def remove_kpt_2d_annotation(self, joint_idx):
        self.joint_anno_dict_l.pop(joint_idx)
        self.joint_anno_dict_r.pop(joint_idx)
        self.update_img_display()
        
    def synch_kpt_2d_annotation(self, side, joint_idx, row, col):
        if self.crop_info is None:
            return
        img_crop_top, img_crop_bot, img_crop_left, img_crop_right, padding_top, \
            padding_bot, padding_left, padding_right, crop_h, crop_w = self.crop_info
        if side == 'left':
            scale = params.IMG_SIZE / crop_h
            row_crop = round((row - img_crop_top + padding_top)*scale)
            col_crop = round((col - img_crop_left + padding_left)*scale)
            self.add_kpt_2d_annotation_right(True, joint_idx, row_crop, col_crop)
        else:
            scale = crop_h / params.IMG_SIZE
            row_orig = round(row*scale + img_crop_top - padding_top)
            col_orig = round(col*scale + img_crop_left - padding_left)
            self.add_kpt_2d_annotation_left(True, joint_idx, row_orig, col_orig)
        
    def update_rendered_img(self):
        self.img_center = self.get_rendered_img()
        self.update_img_display()
        
    # Callback functions
    def on_trackbar_joint(self, val):
        self.joint_selected = int(val)
        if self.is_debugging:
            print('Joint {} selected'.format(self.joint_selected))
        
    def mouse_left_click_callback(self, event):
        row, col = event.y, event.x
        if self.in_img_left_range(row, col):
            # Clicked in left image
            row -= params.IMG_PADDING
            col -= params.IMG_PADDING
            self.add_kpt_2d_annotation_left(False, self.joint_selected, row, col)
        elif self.in_img_right_range(row, col):
            # Clicked in right image
            row -= params.IMG_PADDING
            col -= params.IMG_PADDING*5 + params.IMG_SIZE*2
            self.add_kpt_2d_annotation_right(False, self.joint_selected, row, col)
        else:
            # Remove 2D kpt for current joint
            self.remove_kpt_2d_annotation(self.joint_selected)
            
    def mouse_right_click_callback(self, event):
        row, col = event.y, event.x
        if self.in_img_left_range(row, col):
            row -= params.IMG_PADDING
            col -= params.IMG_PADDING
            self.crop_center = (row, col)
            self.update_crop_box()
            if self.is_debugging:
                print('Crop center set: {}'.format(self.crop_center))
        else:
            self.reset_crop_box()
            self.update_img_display()
            
    def mouse_wheel_callback(self, event):
        if self.crop_center is not None:
            if event.delta > 0:
                self.crop_size += params.CROP_SIZE_TICK_SIZE
                self.crop_size = min(self.crop_size, params.CROP_MAX_SIZE)
            else:
                self.crop_size -= params.CROP_SIZE_TICK_SIZE
                self.crop_size = max(self.crop_size, params.CROP_MIN_SIZE)
            self.update_crop_box()
       
    def button_flip_callback(self):
        if self.args.hand_mode == 'l':
            self.args.hand_mode = 'r'
        else:
            self.args.hand_mode = 'l'
        self.img_left = cv2.flip(self.img_left, 1)
        self.update_img_display()
       
    def button_toggle_callback(self):
        self.img_toggle_state = not self.img_toggle_state
        if self.img_toggle_state == False: 
            self.img_center = self.get_rendered_img()
        else:
            self.img_center = self.img_left
        self.update_img_display()
            
    def button_pred_2d_callback(self):
        if self.model_2d is None:
            print('Model 2D not loaded.')
            return None
        else:
            if self.crop_center is None:
                print('Error, right click to generate bounding box first.')
                return None
            else:
                img_resized = cv2.resize(self.img_right, (params.CROP_SIZE_PRED, params.CROP_SIZE_PRED))
                img_transposed = img_resized.transpose(2, 0, 1).astype(np.float32)
                img_input_tensor = normalize_tensor(torch.from_numpy(img_transposed), 128.0, 256.0)\
                    .unsqueeze_(0).cuda()
                heatmaps_pred = self.model_2d(img_input_tensor)
                heatmaps_np = heatmaps_pred[0].cpu().data.numpy()
                kpts_2d = get_2d_kpts(heatmaps_np, img_h=params.IMG_SIZE, img_w=params.IMG_SIZE, \
                    num_keypoints=params.NUM_KPTS)
                for joint_selected, kpt_2d in enumerate(kpts_2d):
                    row, col = kpt_2d
                    self.add_kpt_2d_annotation_right(False, joint_selected, row, col)
                return kpts_2d
                                
    def button_fit_root_callback(self):
        self.mano_fit_tool.fit_xyz_root_annotate()
        self.update_rendered_img()
        self.update_slider_values()
            
    def button_fit_2d_callback(self):
        self.mano_fit_tool.fit_2d_pose_annotate()
        self.update_rendered_img()
        self.update_slider_values()
        
    def button_fit_3d_3rd_callback(self):
        if self.model_3d_3rd is None:
            print('Model third-person 3D not loaded.')
        else:
            kpts_2d = self.button_pred_2d_callback()
            if kpts_2d is not None:
                self.mano_fit_tool.reset_parameters(keep_mano = True)
                kpts_2d = kpts_2d / params.IMG_SIZE * params.CROP_SIZE_PRED
                heatmaps_np = generate_heatmaps((params.CROP_SIZE_PRED, params.CROP_SIZE_PRED), \
                    params.CROP_STRIDE_PRED, kpts_2d, sigma=params.HEATMAP_SIGMA, is_ratio=False)
                heatmaps_tensor = torch.from_numpy(heatmaps_np.transpose(2, 0, 1)).unsqueeze_(0).cuda()
                kpts_3d_can_pred = self.model_3d_3rd(heatmaps_tensor)
                
                kpts_3d_can_pred_np = kpts_3d_can_pred.cpu().data.numpy()[0].reshape(21, 3)
                kpts_2d_glob_gt_np = self.get_kpts_2d_glob_gt_from_dict()
                # Fit kpts 3d canonical
                self.mano_fit_tool.fit_3d_can_init(kpts_3d_can_pred_np, is_tracking = False)
                # Fit xyz root
                kpts_3d_glob_projected = self.kpts_global_project_tool.canon_to_global\
                    (kpts_2d_glob_gt_np/params.IMG_SIZE, kpts_3d_can_pred_np)
                self.mano_fit_tool.set_xyz_root_with_projection(kpts_3d_glob_projected)
                self.mano_fit_tool.fit_xyz_root_init(kpts_2d_glob_gt_np, is_tracking = False)
                # Fit pose
                self.mano_fit_tool.fit_all_pose(kpts_3d_can_pred_np, kpts_2d_glob_gt_np, is_tracking = False)
                self.update_rendered_img()
                self.update_slider_values()
                
    def button_fit_3d_ego_callback(self):
        if self.model_3d_ego is None:
            print('Model third-person 3D not loaded.')
        else:
            kpts_2d = self.button_pred_2d_callback()
            if kpts_2d is not None:
                # Recreate heatmaps input
                self.mano_fit_tool.reset_parameters(keep_mano = True)
                kpts_2d = kpts_2d / params.IMG_SIZE * params.CROP_SIZE_PRED
                heatmaps_np = generate_heatmaps((params.CROP_SIZE_PRED, params.CROP_SIZE_PRED), \
                    params.CROP_STRIDE_PRED, kpts_2d, sigma=params.HEATMAP_SIGMA, is_ratio=False)
                heatmaps_tensor = torch.from_numpy(heatmaps_np.transpose(2, 0, 1)).unsqueeze_(0).cuda()
                kpts_3d_can_pred = self.model_3d_ego(heatmaps_tensor)
                # Pred 3D canonical pose
                kpts_3d_can_pred_np = kpts_3d_can_pred.cpu().data.numpy()[0].reshape(21, 3)
                kpts_2d_glob_gt_np = self.get_kpts_2d_glob_gt_from_dict()
                # Fit kpts 3d canonical
                self.mano_fit_tool.fit_3d_can_init(kpts_3d_can_pred_np, is_tracking = False)
                # Fit xyz root
                kpts_3d_glob_projected = self.kpts_global_project_tool.canon_to_global\
                    (kpts_2d_glob_gt_np/params.IMG_SIZE, kpts_3d_can_pred_np)
                self.mano_fit_tool.set_xyz_root_with_projection(kpts_3d_glob_projected)
                self.mano_fit_tool.fit_xyz_root_init(kpts_2d_glob_gt_np, is_tracking = False)
                # Fit pose
                self.mano_fit_tool.fit_all_pose(kpts_3d_can_pred_np, kpts_2d_glob_gt_np, is_tracking = False)
                self.update_rendered_img()
                self.update_slider_values()
        
    def button_prev_callback(self):
        self.img_i -= 1
        if self.img_i < 0:
            self.img_i = 0
        self.init_ui()
        
    def button_next_callback(self):
        self.img_i += 1
        if self.img_i >= len(self.img_path_list):
            print('Annotation finished.')
            sys.exit()
        self.init_ui()
        
    def button_save_callback(self):
        self.results_saved = True
        # Save results
        img_path = self.img_path_list[self.img_i]
        mano_save_np = self.mano_fit_tool.get_mano()
        ext = '.' + img_path.split('.')[-1]
        mano_save_np_path = img_path.replace(ext, '_mano_{}.npy'.format(self.args.hand_mode))
        np.save(mano_save_np_path, mano_save_np)
        kpts_2d_glob_np = self.mano_fit_tool.get_kpts_2d_glob()/params.IMG_SIZE
        kpts_2d_glob_np_path = img_path.replace(ext, '_kpts_2d_glob_{}.npy'.format(self.args.hand_mode))
        np.save(kpts_2d_glob_np_path, kpts_2d_glob_np)
        kpts_3d_glob_np = self.mano_fit_tool.get_kpts_3d_glob()
        kpts_3d_glob_np_path = img_path.replace(ext, '_kpts_3d_glob_{}.npy'.format(self.args.hand_mode))
        np.save(kpts_3d_glob_np_path, kpts_3d_glob_np)
        print('Results saved at {}'.format(mano_save_np_path))
        
    def button_reset_callback(self):
        self.init_variables()
        self.init_frame_info()
            
    def on_trackbar_mano_rot0(self, val):
        self.mano_fit_tool.set_input_rot_i(params.rot0_idx, get_mano_val_from_slider('glob', 0, int(val)))
        self.update_rendered_img()

    def on_trackbar_mano_rot1(self, val):
        self.mano_fit_tool.set_input_rot_i(params.rot1_idx, get_mano_val_from_slider('glob', 1, int(val)))
        self.update_rendered_img()

    def on_trackbar_mano_rot2(self, val):
        self.mano_fit_tool.set_input_rot_i(params.rot2_idx, get_mano_val_from_slider('glob', 2, int(val)))
        self.update_rendered_img()
        
    def button_freeze_rot_callback(self):
        current_freeze_state = self.mano_fit_tool.toggle_rot_freeze_state()
        self.toggle_freeze_restore_text(self.window.freeze_text_rot_variable, current_freeze_state)
        self.update_slider_rot()
        
    def on_trackbar_mano_xyz_root0(self, val):
        self.mano_fit_tool.set_xyz_root_i(0, get_mano_val_from_slider('xyz_root', 0, int(val)))
        self.update_rendered_img()
    
    def on_trackbar_mano_xyz_root1(self, val):
        self.mano_fit_tool.set_xyz_root_i(1, get_mano_val_from_slider('xyz_root', 1, int(val)))
        self.update_rendered_img()
        
    def on_trackbar_mano_xyz_root2(self, val):
        self.mano_fit_tool.set_xyz_root_i(2, get_mano_val_from_slider('xyz_root', 2, int(val)))
        self.update_rendered_img()
        
    def on_trackbar_mano_finger0_ver(self, val):
        self.mano_fit_tool.set_input_pose_i(params.fin0_ver_idx1 - 3, get_mano_val_from_slider('finger0', 0, int(val)))
        self.mano_fit_tool.set_input_pose_i(params.fin0_ver_idx2 - 3, get_mano_val_from_slider('finger0', 1, int(val)))
        self.mano_fit_tool.set_input_pose_i(params.fin0_ver_idx3 - 3, get_mano_val_from_slider('finger0', 2, int(val)))
        self.update_slider_values_fin0()
        self.update_rendered_img()

    def on_trackbar_mano_finger0_ver0(self, val):
        self.mano_fit_tool.set_input_pose_i(params.fin0_ver_idx1 - 3, get_mano_val_from_slider('finger0', 0, int(val)))
        self.update_slider_values_fin0()
        self.update_rendered_img()
        
    def on_trackbar_mano_finger0_ver1(self, val):
        self.mano_fit_tool.set_input_pose_i(params.fin0_ver_idx2 - 3, get_mano_val_from_slider('finger0', 1, int(val)))
        self.update_rendered_img()
        
    def on_trackbar_mano_finger0_ver2(self, val):
        self.mano_fit_tool.set_input_pose_i(params.fin0_ver_idx3 - 3, get_mano_val_from_slider('finger0', 2, int(val)))
        self.update_rendered_img()

    def on_trackbar_mano_finger0_hor(self, val):
        self.mano_fit_tool.set_input_pose_i(params.fin0_hor_idx1 - 3, get_mano_val_from_slider('finger0', 3, int(val)))
        self.update_rendered_img()

    def on_trackbar_mano_finger0_rot(self, val):
        self.mano_fit_tool.set_input_pose_i(params.fin0_rot_idx1 - 3, get_mano_val_from_slider('finger0', 4, int(val)))
        self.update_rendered_img()
        
    def button_freeze_fin0_callback(self):
        current_freeze_state = self.mano_fit_tool.toggle_finger_freeze_state(0)
        self.toggle_freeze_finger_text(self.window.freeze_text_fin0_variable, current_freeze_state)
        self.update_slider_values_fin0()
        
    def on_trackbar_mano_finger1_ver(self, val):
        self.mano_fit_tool.set_input_pose_i(params.fin1_ver_idx1 - 3, get_mano_val_from_slider('finger1', 0, int(val)))
        self.mano_fit_tool.set_input_pose_i(params.fin1_ver_idx2 - 3, get_mano_val_from_slider('finger1', 1, int(val)))
        self.mano_fit_tool.set_input_pose_i(params.fin1_ver_idx3 - 3, get_mano_val_from_slider('finger1', 2, int(val)))
        self.update_slider_values_fin1()
        self.update_rendered_img()

    def on_trackbar_mano_finger1_ver0(self, val):
        self.mano_fit_tool.set_input_pose_i(params.fin1_ver_idx1 - 3, get_mano_val_from_slider('finger1', 0, int(val)))
        self.update_slider_values_fin1()
        self.update_rendered_img()
        
    def on_trackbar_mano_finger1_ver1(self, val):
        self.mano_fit_tool.set_input_pose_i(params.fin1_ver_idx2 - 3, get_mano_val_from_slider('finger1', 1, int(val)))
        self.update_rendered_img()
        
    def on_trackbar_mano_finger1_ver2(self, val):
        self.mano_fit_tool.set_input_pose_i(params.fin1_ver_idx3 - 3, get_mano_val_from_slider('finger1', 2, int(val)))
        self.update_rendered_img()

    def on_trackbar_mano_finger1_hor(self, val):
        self.mano_fit_tool.set_input_pose_i(params.fin1_hor_idx1 - 3, get_mano_val_from_slider('finger1', 3, int(val)))
        self.update_rendered_img()

    def on_trackbar_mano_finger1_rot(self, val):
        self.mano_fit_tool.set_input_pose_i(params.fin1_rot_idx1 - 3, get_mano_val_from_slider('finger1', 4, int(val)))
        self.update_rendered_img()
        
    def button_freeze_fin1_callback(self):
        current_freeze_state = self.mano_fit_tool.toggle_finger_freeze_state(1)
        self.toggle_freeze_finger_text(self.window.freeze_text_fin1_variable, current_freeze_state)
        self.update_slider_values_fin1()
        
    def on_trackbar_mano_finger2_ver(self, val):
        self.mano_fit_tool.set_input_pose_i(params.fin2_ver_idx1 - 3, get_mano_val_from_slider('finger2', 0, int(val)))
        self.mano_fit_tool.set_input_pose_i(params.fin2_ver_idx2 - 3, get_mano_val_from_slider('finger2', 1, int(val)))
        self.mano_fit_tool.set_input_pose_i(params.fin2_ver_idx3 - 3, get_mano_val_from_slider('finger2', 2, int(val)))
        self.update_slider_values_fin2()
        self.update_rendered_img()     

    def on_trackbar_mano_finger2_ver0(self, val):
        self.mano_fit_tool.set_input_pose_i(params.fin2_ver_idx1 - 3, get_mano_val_from_slider('finger2', 0, int(val)))
        self.update_slider_values_fin2()
        self.update_rendered_img()
        
    def on_trackbar_mano_finger2_ver1(self, val):
        self.mano_fit_tool.set_input_pose_i(params.fin2_ver_idx2 - 3, get_mano_val_from_slider('finger2', 1, int(val)))
        self.update_rendered_img()
        
    def on_trackbar_mano_finger2_ver2(self, val):
        self.mano_fit_tool.set_input_pose_i(params.fin2_ver_idx3 - 3, get_mano_val_from_slider('finger2', 2, int(val)))
        self.update_rendered_img()

    def on_trackbar_mano_finger2_hor(self, val):
        self.mano_fit_tool.set_input_pose_i(params.fin2_hor_idx1 - 3, get_mano_val_from_slider('finger2', 3, int(val)))
        self.update_rendered_img()

    def on_trackbar_mano_finger2_rot(self, val):
        self.mano_fit_tool.set_input_pose_i(params.fin2_rot_idx1 - 3, get_mano_val_from_slider('finger2', 4, int(val)))
        self.update_rendered_img()
        
    def button_freeze_fin2_callback(self):
        current_freeze_state = self.mano_fit_tool.toggle_finger_freeze_state(2)
        self.toggle_freeze_finger_text(self.window.freeze_text_fin2_variable, current_freeze_state)
        self.update_slider_values_fin2()
        
    def on_trackbar_mano_finger3_ver(self, val):
        self.mano_fit_tool.set_input_pose_i(params.fin3_ver_idx1 - 3, get_mano_val_from_slider('finger3', 0, int(val)))
        self.mano_fit_tool.set_input_pose_i(params.fin3_ver_idx2 - 3, get_mano_val_from_slider('finger3', 1, int(val)))
        self.mano_fit_tool.set_input_pose_i(params.fin3_ver_idx3 - 3, get_mano_val_from_slider('finger3', 2, int(val)))
        self.update_slider_values_fin3()
        self.update_rendered_img()
        
    def on_trackbar_mano_finger3_ver0(self, val):
        self.mano_fit_tool.set_input_pose_i(params.fin3_ver_idx1 - 3, get_mano_val_from_slider('finger3', 0, int(val)))
        self.update_slider_values_fin3()
        self.update_rendered_img()
        
    def on_trackbar_mano_finger3_ver1(self, val):
        self.mano_fit_tool.set_input_pose_i(params.fin3_ver_idx2 - 3, get_mano_val_from_slider('finger3', 1, int(val)))
        self.update_rendered_img()
        
    def on_trackbar_mano_finger3_ver2(self, val):
        self.mano_fit_tool.set_input_pose_i(params.fin3_ver_idx3 - 3, get_mano_val_from_slider('finger3', 2, int(val)))
        self.update_rendered_img()

    def on_trackbar_mano_finger3_hor(self, val):
        self.mano_fit_tool.set_input_pose_i(params.fin3_hor_idx1 - 3, get_mano_val_from_slider('finger3', 3, int(val)))
        self.update_rendered_img()

    def on_trackbar_mano_finger3_rot(self, val):
        self.mano_fit_tool.set_input_pose_i(params.fin3_rot_idx1 - 3, get_mano_val_from_slider('finger3', 4, int(val)))
        self.update_rendered_img()
        
    def button_freeze_fin3_callback(self):
        current_freeze_state = self.mano_fit_tool.toggle_finger_freeze_state(3)
        self.toggle_freeze_finger_text(self.window.freeze_text_fin3_variable, current_freeze_state)
        self.update_slider_values_fin3()
        
    def on_trackbar_mano_finger4_ver(self, val):
        self.mano_fit_tool.set_input_pose_i(params.fin4_ver_idx1 - 3, get_mano_val_from_slider('finger4', 0, int(val)))
        self.mano_fit_tool.set_input_pose_i(params.fin4_ver_idx2 - 3, get_mano_val_from_slider('finger4', 1, int(val)))
        self.mano_fit_tool.set_input_pose_i(params.fin4_ver_idx3 - 3, get_mano_val_from_slider('finger4', 2, int(val)))
        # For proper pinky vertical movement
        self.mano_fit_tool.set_input_pose_i(params.fin4_ver_fix_idx - 3, -params.mano_fin4_ver_range2[int(val)])
        self.update_slider_values_fin4()
        self.update_rendered_img()
        
    def on_trackbar_mano_finger4_ver0(self, val):
        self.mano_fit_tool.set_input_pose_i(params.fin4_ver_idx1 - 3, get_mano_val_from_slider('finger4', 0, int(val)))
        # For proper pinky vertical movement
        self.mano_fit_tool.set_input_pose_i(params.fin4_ver_fix_idx - 3, -params.mano_fin4_ver_range2[int(val)])
        self.update_slider_values_fin4()
        self.update_rendered_img()
        
    def on_trackbar_mano_finger4_ver1(self, val):
        self.mano_fit_tool.set_input_pose_i(params.fin4_ver_idx2 - 3, get_mano_val_from_slider('finger4', 1, int(val)))
        # For proper pinky vertical movement
        self.mano_fit_tool.set_input_pose_i(params.fin4_ver_fix_idx - 3, -params.mano_fin4_ver_range2[int(val)])
        self.update_rendered_img()
        
    def on_trackbar_mano_finger4_ver2(self, val):
        self.mano_fit_tool.set_input_pose_i(params.fin4_ver_idx3 - 3, get_mano_val_from_slider('finger4', 2, int(val)))
        # For proper pinky vertical movement
        self.mano_fit_tool.set_input_pose_i(params.fin4_ver_fix_idx - 3, -params.mano_fin4_ver_range2[int(val)])
        self.update_rendered_img()

    def on_trackbar_mano_finger4_hor(self, val):
        self.mano_fit_tool.set_input_pose_i(params.fin4_hor_idx1 - 3, get_mano_val_from_slider('finger4', 3, int(val)))
        self.update_rendered_img()

    def on_trackbar_mano_finger4_rot(self, val):
        self.mano_fit_tool.set_input_pose_i(params.fin4_rot_idx1 - 3, get_mano_val_from_slider('finger4', 4, int(val)))
        self.update_rendered_img()
        
    def button_freeze_fin4_callback(self):
        current_freeze_state = self.mano_fit_tool.toggle_finger_freeze_state(4)
        self.toggle_freeze_finger_text(self.window.freeze_text_fin4_variable, current_freeze_state)
        self.update_slider_values_fin4()
        
def parse():
    parser = argparse.ArgumentParser()
    # [Optional] Add arguments here
    return parser.parse_args()

if __name__ == '__main__':
    args = parse()
    annotator_ui_app = pose_annotation_app(args)
    annotator_ui_app.run_app()