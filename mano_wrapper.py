import params
import torch
from models import mano_layer_annotate

from pytorch3d.renderer import (
    OpenGLPerspectiveCameras, look_at_view_transform, look_at_rotation, 
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, PointLights
)
from utils import *
from models import mano_layer_render

def init_pytorch3d(is_renderer = False):
    device = torch.device("cuda:0")
    # Initialize an OpenGL perspective camera. 
    cameras = OpenGLPerspectiveCameras(fov = 60.0, device=device)
    # Set blend params
    blend_params = BlendParams(sigma=1e-4, gamma=1e-4)
    # Define the settings for rasterization and shading.
    if is_renderer:
        image_size = params.pytorch3d_img_w*2
    else:
        image_size = params.pytorch3d_img_w
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma, 
        faces_per_pixel=20, 
        bin_size = None,  
        max_faces_per_bin = None
    )
    silhouette_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        ),
        shader=SoftSilhouetteShader(blend_params=blend_params)
    )
    lights = PointLights(device=device, location=((2.0, 2.0, -2.0),))
    phong_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        ),
        shader=HardPhongShader(device=device, cameras=cameras, lights=lights)
    )
    return device, cameras, blend_params, raster_settings, silhouette_renderer, lights, phong_renderer

def load_mano_model(mano_path, silhouette_renderer, phong_renderer, device, is_annotating):
    root_idx = 9 if not is_annotating else 0
    mano_model = mano_layer_annotate.Model(
        mano_path = mano_path, 
        renderer = silhouette_renderer, 
        renderer2 = phong_renderer,
        device = device, 
        batch_size = 1,
        root_idx = root_idx).to(device)
    return mano_model
    
def load_mano_renderer(mano_path, silhouette_renderer, phong_renderer, device):
    mano_model = mano_layer_render.Model(
        mano_path = mano_path, 
        renderer = phong_renderer,
        device = device, 
        batch_size = 2, 
        root_idx = 0).to(device)
    return mano_model
    
def set_optimizer_lr(optimizer, lr_new, is_debugging):
    if is_debugging:
        print("Setting lr: ", lr_new)
    for g in optimizer.param_groups:
        g['lr'] = lr_new

class mano_renderer:
    def __init__(self, mano_path):    
        # Prep pytorch 3d for visualization
        self.device, self.cameras, self.blend_params, self.raster_settings, self.silhouette_renderer, \
            self.lights, self.phong_renderer = init_pytorch3d(is_renderer = True)
        self.mano_model = load_mano_renderer(mano_path, self.silhouette_renderer, self.phong_renderer, self.device)
        self.kpts_2d_l, self.kpts_2d_r = None, None
    
    def set_two_hand_verts(self, verts0, verts1):
        self.mano_model.set_verts(verts0, verts1)
        
    def set_two_hand_kpts(self, kpts_2d_l, kpts_2d_r):
        self.kpts_2d_l = kpts_2d_l
        if kpts_2d_l is not None:
            self.kpts_2d_l[:,1] = 1.0 - self.kpts_2d_l[:,1]
            self.kpts_2d_l = self.kpts_2d_l*params.pytorch3d_img_w*2
        self.kpts_2d_r = kpts_2d_r
        if kpts_2d_r is not None:
            self.kpts_2d_r = kpts_2d_r*params.pytorch3d_img_w*2
        
    def get_rendered_img(self, kpts_2d_glob = None):
        img_rendered = self.mano_model()
        if kpts_2d_glob is not None:
            kpts_2d_glob = kpts_2d_glob * params.pytorch3d_img_w*2
            img_rendered = paint_kpts(None, img_rendered, kpts_2d_glob[0].cpu().data.numpy())
        else:
            if self.kpts_2d_l is not None:
                img_rendered = paint_kpts(None, img_rendered, self.kpts_2d_l)
            if self.kpts_2d_r is not None:
                img_rendered = paint_kpts(None, img_rendered, self.kpts_2d_r)
        return img_rendered

class mano_fitter:
    def __init__(self, mano_path, is_annotating = False):    
        self.lr_rot_init = params.lr_rot_init
        self.lr_pose_init = params.lr_pose_init
        self.lr_xyz_root_init = params.lr_xyz_root_init
        self.lr_all_init = params.lr_all_init
        self.loss_rot_best = float('inf')
        self.loss_pose_best = float('inf')
        self.loss_xyz_root_best = float('inf')
        self.loss_all_best = float('inf')
        self.img_input_size = None

        # Prep pytorch 3d for visualization
        self.device, self.cameras, self.blend_params, self.raster_settings, self.silhouette_renderer, \
            self.lights, self.phong_renderer = init_pytorch3d()
        self.mano_model = load_mano_model(mano_path, self.silhouette_renderer, self.phong_renderer, \
            self.device, is_annotating)
        self.mano_model.change_render_setting(True)
        
        self.optimizer_adam_mano_fit_all = None
        self.lr_scheduler_all = None
        
    def set_input_size(self, img_input_size):
        self.img_input_size = img_input_size
        
    def set_shape(self, mano_shape):
        self.mano_model.set_input_shape(torch.from_numpy(mano_shape).cuda())
        
    def get_mano(self):
        xy_root = self.mano_model.xy_root.cpu().data.numpy().reshape(-1)
        z_root = self.mano_model.z_root.cpu().data.numpy().reshape(-1)
        input_rot = self.mano_model.input_rot.cpu().data.numpy().reshape(-1)
        input_pose = self.mano_model.input_pose.cpu().data.numpy().reshape(-1)
        mano_np = np.concatenate((xy_root, z_root, input_rot, input_pose))
        return mano_np
        
    def set_mano(self, mano_np, root_idx = None):
        xyz_tensor = torch.from_numpy(mano_np[:3]).cuda()
        input_rot_tensor = torch.from_numpy(mano_np[3:6]).cuda()
        input_pose_tensor = torch.from_numpy(mano_np[6:]).cuda()
        self.mano_model.set_xyz_root(xyz_tensor)
        self.mano_model.set_input_rot(input_rot_tensor)
        self.mano_model.set_input_pose(input_pose_tensor)
        if root_idx is not None:
            hand_joints, _ = self.mano_model.forward_basic()
            xyz_root_new = hand_joints[0,0] - (hand_joints[0,root_idx] - hand_joints[0,0])
            self.mano_model.set_xyz_root(xyz_root_new)
        
    def set_xyz_root_with_projection(self, kpts_3d_glob_projected):
        self.mano_model.set_xyz_root(torch.from_numpy(kpts_3d_glob_projected[params.hand_root_joint, \
            [1, 0, 2]]).cuda())
        
    def set_input_rot(self, input_rot_tensor):
        self.mano_model.set_input_rot(input_rot_tensor)
        
    def set_input_rot_i(self, rot_i, rot_val):
        self.mano_model.input_rot[0, rot_i] = rot_val
        
    def get_input_rot_i(self, rot_i):
        return self.mano_model.input_rot[0, rot_i]
        
    def toggle_rot_freeze_state(self):
        return self.mano_model.toggle_rot_freeze()
        
    def set_xyz_root_i(self, xyz_i, val):
        if xyz_i < 2:
            self.mano_model.xy_root[0, xyz_i] = val
        else:
            self.mano_model.z_root[0, 0] = val
            
    def get_xyz_root_i(self, xyz_i):
        if xyz_i < 2:
            return self.mano_model.xy_root[0, xyz_i]
        else:
            return self.mano_model.z_root[0, 0]
            
    def set_input_pose_i(self, pose_i, pose_val):
        self.mano_model.input_pose[0, pose_i] = pose_val
        
    def get_input_pose_i(self, pose_i):
        return self.mano_model.input_pose[0, pose_i]
        
    def toggle_finger_freeze_state(self, finger_i):
        return self.mano_model.toggle_finger_freeze(finger_i)
        
    def set_kpt_2d(self, joint_idx, kpt_2d_glob):
        self.mano_model.set_kpts_2d_glob_gt_val(joint_idx, kpt_2d_glob)
        
    def get_kpts_2d_glob(self):
        self.mano_model.forward_basic()
        return self.mano_model.kpts_2d_glob.cpu().data.numpy().reshape(-1, 2)
        
    def get_kpts_3d_glob(self):
        self.mano_model.forward_basic()
        return self.mano_model.kpts_3d_glob.cpu().data.numpy().reshape(-1, 3)
        
    def fit_3d_can_init(self, kpts_3d_can, is_tracking):
        # Set 3D target
        kpts_3d_can = kpts_3d_can.reshape(-1, 3)
        kpts_3d_can -= kpts_3d_can[params.hand_root_joint]
        kpts_3d_glob = kpts_3d_can * params.mano_key_bone_len
        self.mano_model.set_kpts_3d_glob_leap_no_palm(kpts_3d_glob)
        
        is_debugging = True
        # Step1: fit canonical pose using canonical 3D kpts
        self.mano_model.change_rot_grads(True)
        self.mano_model.set_rot_only()
        optimizer_adam_mano_fit, lr_scheduler = self.reset_mano_optimizer('rot')
        self.fit_mano(optimizer_adam_mano_fit, lr_scheduler, "rot", 50, is_loss_leap3d=True, is_optimizing=True, \
            is_debugging=is_debugging, is_tracking=is_tracking) 
        self.reset_mano_optimization_var()
        
        self.mano_model.change_rot_grads(True)
        self.mano_model.change_pose_grads(True)
        optimizer_adam_mano_fit, lr_scheduler = self.reset_mano_optimizer('pose')
        self.fit_mano(optimizer_adam_mano_fit, lr_scheduler, "pose", 50, is_loss_leap3d=True, is_loss_reg=True, \
            is_optimizing=True, is_debugging=is_debugging, is_tracking=is_tracking)
        self.reset_mano_optimization_var()
        
    def fit_xyz_root_init(self, kpts_2d_glob, is_tracking):
        # Scale kpts 2d glob to match pytorch 3d resolution
        assert self.img_input_size is not None
        kpts_2d_glob = kpts_2d_glob*np.array([float(params.pytorch3d_img_w)/self.img_input_size, \
            float(params.pytorch3d_img_w)/self.img_input_size])
        
        is_debugging = True
        
        # Step2: fit xyz root using global 2D kpts
        self.mano_model.set_xyz_root(torch.tensor([0, 0, 50.0]).view(1, -1).cuda())
        joint_idx_set = set([0, 1, 5, 9, 13, 17])
        for joint_idx, kpt_2d_glob in enumerate(kpts_2d_glob):
            if joint_idx in joint_idx_set:
                self.mano_model.set_kpts_2d_glob_gt_val(joint_idx, kpt_2d_glob)
        self.mano_model.change_root_grads(True)
        optimizer_adam_mano_fit_xyz_root, lr_scheduler = self.reset_mano_optimizer('xyz_root')
        self.fit_mano(optimizer_adam_mano_fit_xyz_root, lr_scheduler, "xyz_root", 50, is_loss_2d_glob=True, \
            is_optimizing=True, is_debugging=is_debugging, is_tracking=is_tracking)
        self.reset_mano_optimization_var()
        
    def fit_all_pose(self, kpts_3d_can, kpts_2d_glob, is_tracking):
        # Step final: fit all params using global 2D and canonical 3D kpts
        # Set 3D target
        kpts_3d_can = kpts_3d_can.reshape(-1, 3)
        kpts_3d_can -= kpts_3d_can[params.hand_root_joint]
        kpts_3d_glob = kpts_3d_can * params.mano_key_bone_len
        self.mano_model.set_kpts_3d_glob_leap_no_palm(kpts_3d_glob, with_xyz_root = True)
        
        is_debugging = True
        
        for joint_idx, kpt_2d_glob in enumerate(kpts_2d_glob):
            self.mano_model.set_kpts_2d_glob_gt_val(joint_idx, kpt_2d_glob)
        self.mano_model.change_root_grads(True)
        self.mano_model.change_rot_grads(True)
        self.mano_model.change_pose_grads(True)
        if self.optimizer_adam_mano_fit_all is None:
            self.optimizer_adam_mano_fit_all, self.lr_scheduler_all = self.reset_mano_optimizer('all')
        num_iters = 50 if not is_tracking else 10
        self.fit_mano(self.optimizer_adam_mano_fit_all, self.lr_scheduler_all, "all", num_iters, \
            is_loss_2d_glob=True, is_loss_leap3d=True, is_loss_reg=True, is_optimizing=True, is_debugging=is_debugging, is_tracking=is_tracking)
        self.reset_mano_optimization_var()
        
    def fit_can_pose(self, kpts_3d_can, kpts_2d_glob, is_tracking):
        # Scale kpts 2d glob to match pytorch 3d resolution
        assert self.img_input_size is not None
        kpts_2d_glob = kpts_2d_glob*np.array([float(params.pytorch3d_img_w)/self.img_input_size, \
            float(params.pytorch3d_img_w)/self.img_input_size])
    
        # Set target
        kpts_3d_can = kpts_3d_can.reshape(-1, 3)
        kpts_3d_can -= kpts_3d_can[params.hand_root_joint]
        kpts_3d_glob = kpts_3d_can * params.mano_key_bone_len
        self.mano_model.set_kpts_3d_glob_leap_no_palm(kpts_3d_glob)
        
        is_debugging = False
        # Step1: fit canonical pose using canonical 3D kpts
        self.mano_model.change_rot_grads(True)
        self.mano_model.set_rot_only()
        optimizer_adam_mano_fit, lr_scheduler = self.reset_mano_optimizer('rot')
        set_optimizer_lr(optimizer_adam_mano_fit, self.lr_rot_init, is_debugging)
        self.fit_mano(optimizer_adam_mano_fit, lr_scheduler, "rot", 100, is_loss_leap3d=True, \
            is_optimizing=True, is_debugging=is_debugging, is_tracking=is_tracking) 
        self.reset_mano_optimization_var()
        
        self.mano_model.change_rot_grads(True)
        self.mano_model.change_pose_grads(True)
        optimizer_adam_mano_fit, lr_scheduler = self.reset_mano_optimizer('pose')
        set_optimizer_lr(optimizer_adam_mano_fit, self.lr_pose_init, is_debugging)
        self.fit_mano(optimizer_adam_mano_fit, lr_scheduler, "pose", 100, is_loss_leap3d=True, \
            is_loss_reg=True, is_optimizing=True, is_debugging=is_debugging, is_tracking=is_tracking)
        self.reset_mano_optimization_var()
        
        # Step2: fit xyz root using global 2D kpts
        self.mano_model.set_xyz_root(torch.tensor([0, 0, 50.0]).view(1, -1).cuda())
        joint_idx_set = set([0, 1, 5, 9, 13, 17])
        for joint_idx, kpt_2d_glob in enumerate(kpts_2d_glob):
            if joint_idx in joint_idx_set:
                self.mano_model.set_kpts_2d_glob_gt_val(joint_idx, kpt_2d_glob)
        self.mano_model.change_root_grads(True)
        optimizer_adam_mano_fit, lr_scheduler = self.reset_mano_optimizer('xyz_root')
        set_optimizer_lr(optimizer_adam_mano_fit, self.lr_xyz_root_init, is_debugging)
        self.fit_mano(optimizer_adam_mano_fit, lr_scheduler, "xyz_root", 100, is_loss_2d_glob=True, \
            is_optimizing=True, is_debugging=is_debugging, is_tracking=is_tracking)
        self.reset_mano_optimization_var()
        
        self.mano_model.change_render_setting(True)
        _, _, _, _, _, _, img_render1 = self.mano_model()
        img_render1 = (img_render1.cpu().data.numpy()[0][:,:,:4]*255.0).astype(np.uint8)
        img_render1 = paint_kpts(None, img_render1, self.mano_model.kpts_2d_glob[0].cpu().data.numpy())
        
        # Step4: fit all params using global 2D and canonical 3D kpts
        for joint_idx, kpt_2d_glob in enumerate(kpts_2d_glob):
            self.mano_model.set_kpts_2d_glob_gt_val(joint_idx, kpt_2d_glob)
        self.mano_model.change_root_grads(True)
        self.mano_model.change_rot_grads(True)
        self.mano_model.change_pose_grads(True)
        optimizer_adam_mano_fit, lr_scheduler = self.reset_mano_optimizer('all')
        set_optimizer_lr(optimizer_adam_mano_fit, self.lr_all_init, is_debugging)
        self.fit_mano(optimizer_adam_mano_fit, lr_scheduler, "all", 300, is_loss_2d_glob=True, \
            is_loss_reg=True, is_optimizing=True, is_debugging=is_debugging, is_tracking=is_tracking)
        self.reset_mano_optimization_var()
        
        self.mano_model.change_render_setting(True)
        _, _, _, _, _, _, img_render2 = self.mano_model()
        img_render2 = (img_render2.cpu().data.numpy()[0][:,:,:4]*255.0).astype(np.uint8)
        img_render2 = paint_kpts(None, img_render2, self.mano_model.kpts_2d_glob[0].cpu().data.numpy())
        return img_render1, img_render2
        
    def fit_xyz_root_annotate(self):
        is_debugging = False
        self.mano_model.set_root_only()
        self.mano_model.change_root_grads(True)
        optimizer_adam_mano_fit, lr_scheduler = self.reset_mano_optimizer('xyz_root', is_annotating = True)
        self.fit_mano(optimizer_adam_mano_fit, lr_scheduler, "xyz_root", 100, is_loss_2d_glob=True, \
            is_optimizing=True, is_debugging=is_debugging, is_tracking=False)
        self.reset_mano_optimization_var()
        
    def fit_2d_pose_annotate(self):
        is_debugging = False
        self.mano_model.change_root_grads(True)
        self.mano_model.change_rot_grads(True)
        self.mano_model.change_pose_grads(True)
        optimizer_adam_mano_fit, lr_scheduler = self.reset_mano_optimizer('pose', is_annotating = True)
        self.fit_mano(optimizer_adam_mano_fit, lr_scheduler, "all", 100, is_loss_2d_glob=True, \
            is_loss_reg=True, is_optimizing=True, is_debugging=is_debugging, is_tracking=False)
        self.reset_mano_optimization_var()
        
    def reset_mano_optimization_var(self):
        self.mano_model.reset_param_grads()
        self.lr_rot_init = params.lr_rot_init
        self.lr_pose_init = params.lr_pose_init
        self.lr_xyz_root_init = params.lr_xyz_root_init
        self.lr_all_init = params.lr_all_init
        self.loss_rot_best = float('inf')
        self.loss_pose_best = float('inf')
        self.loss_xyz_root_best = float('inf')
        self.loss_all_best = float('inf')
                
    def reset_mano_optimizer(self, mode, is_annotating = False):    
        if mode == 'rot':
            if not is_annotating:
                lr_init = params.lr_rot_init
            else:
                lr_init = params.lr_rot_init
            model_params = self.mano_model.parameters()
        elif mode == 'pose':
            if not is_annotating:
                lr_init = params.lr_pose_init
            else:
                lr_init = params.lr_pose_init
            model_params = self.mano_model.parameters()
        elif mode == 'xyz_root':
            if not is_annotating:
                lr_init = params.lr_xyz_root_init
            else:
                lr_init = params.lr_xyz_root_init
            params_dict = dict(self.mano_model.named_parameters())
            lr1 = []
            lr2 = []
            for key, value in params_dict.items():
                if value.requires_grad:
                    if 'xy_root' in key:
                        lr1.append(value)
                    elif 'z_root' in key:
                        lr2.append(value)
            model_params = [{'params': lr1, 'lr': lr_init},
                {'params': lr2, 'lr': lr_init}]
        elif mode == 'all':
            if not is_annotating:
                lr_init = params.lr_all_init
            else:
                lr_init = params.lr_all_init
            params_dict = dict(self.mano_model.named_parameters())
            lr_xyz_root = []
            lr_rot = []
            lr_pose = []
            for key, value in params_dict.items():
                if value.requires_grad:
                    if 'xy_root' in key:
                        lr_xyz_root.append(value)
                    elif 'z_root' in key:
                        lr_xyz_root.append(value)
                    elif 'input_rot' in key:
                        lr_rot.append(value)
                    elif 'input_pose' in key:
                        lr_pose.append(value)
            model_params = [{'params': lr_xyz_root, 'lr': 0.5},
                {'params': lr_rot, 'lr': 0.05},
                {'params': lr_pose, 'lr': 0.05}]
        optimizer_adam_mano_fit = torch.optim.Adam(model_params, lr=lr_init)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_adam_mano_fit, step_size = 1, gamma = 0.5)
        return optimizer_adam_mano_fit, lr_scheduler
            
    def fit_mano(self, optimizer, lr_scheduler, mode, iter_fit, is_loss_seg=False, \
        is_loss_2d_glob=False, is_loss_leap3d=False, is_loss_reg=False, is_optimizing=True, \
        best_performance=True, is_debugging=True, is_tracking=False, is_visualizing=False, show_progress=False):
        self.mano_model.set_loss_mode(is_loss_seg=is_loss_seg, is_loss_2d_glob=is_loss_2d_glob,  
            is_loss_leap3d=is_loss_leap3d, is_loss_reg=is_loss_reg)
        self.mano_model.change_render_setting(False)
            
        if is_debugging:
            print("Fitting {}".format(mode))
        iter_count = 0
        lr_stage = 3
        update_finished = False
        
        if not best_performance:
            rot_th = 20000
            pose_th = 20000
            xyz_root_th = 2000
        else:
            rot_th = 20000
            pose_th = 20000
            xyz_root_th = 1000
        
        while not update_finished and is_optimizing:
            loss_seg_batch, loss_2d_glob_batch, loss_reg_batch, loss_leap3d_batch, _, _ = self.mano_model()

            loss_total = 0
            loss_seg_sum = 0
            loss_2d_glob_sum = 0
            loss_3d_can_sum = 0
            loss_reg_sum = 0
            loss_leap3d_sum = 0
            if is_loss_seg:
                loss_seg_sum = torch.sum(loss_seg_batch)
                loss_total += loss_seg_sum
            if is_loss_2d_glob:
                loss_2d_glob_sum = torch.sum(loss_2d_glob_batch)
                loss_total += loss_2d_glob_sum
            if is_loss_reg:
                loss_reg_sum = torch.sum(loss_reg_batch)
                loss_total += loss_reg_sum
            if is_loss_leap3d:
                loss_leap3d_sum = 100000*torch.sum(loss_leap3d_batch)
                loss_total += loss_leap3d_sum

            if is_debugging and iter_count % 1 == 0:  
                print("[{}] Fit loss total {:.5f}, loss 2d {:.2f}, loss 3d {:.5f}, loss leap3d {:.2f}, loss reg {:.2f}, "\
                    .format(iter_count, float(loss_total), float(loss_2d_glob_sum), float(loss_3d_can_sum), \
                    float(loss_leap3d_sum), float(loss_reg_sum)))
                
            iter_count += 1
        
            # Check stopping criteria
            if is_tracking:
                if mode == 'rot':
                    if loss_leap3d_sum < rot_th:
                        update_finished = True
                        
                if mode == 'pose':
                    if loss_leap3d_sum < pose_th:
                        update_finished = True
                        
                if mode == 'xyz_root':
                    if loss_2d_glob_sum < xyz_root_th:
                        update_finished = True
                        
                if mode == 'all':
                    if not best_performance:
                        if iter_count >= 10:
                            if loss_2d_glob_sum < 1500:
                                update_finished = True
                        else:
                            if loss_2d_glob_sum < 1000:
                                update_finished = True
                    else:
                        if loss_2d_glob_sum < 1000:
                            update_finished = True
            
            if iter_count >= iter_fit:
                update_finished = True
                    
            if is_optimizing:
                optimizer.zero_grad()
                loss_total.backward(retain_graph=True)
                optimizer.step()
                
                # Adjust pinky
                with torch.no_grad():
                    self.mano_model.input_pose[0, params.fin4_ver_fix_idx - 3] = -self.mano_model.input_pose\
                        [0, params.fin4_ver_idx2 - 3]
            else:
                update_finished = True

        if is_debugging:
            print("Optimization stopped. Iter = {}".format(iter_count))
        
        if is_visualizing or show_progress:
            self.mano_model.change_render_setting(True)
            _, _, _, _, _, img_render = self.mano_model()
            img_render = (img_render.cpu().data.numpy()[0][:,:,:3]*255.0).astype(np.uint8)
            img_render = paint_kpts(None, img_render, self.mano_model.kpts_2d_glob[0].cpu().data.numpy())
            return img_render
        else:
            self.mano_model()
    
    def get_rendered_img(self):
        self.mano_model.change_render_setting(True)
        _, _, _, _, _, img_rendered = self.mano_model()
        img_rendered = (img_rendered.cpu().data.numpy()[0][:,:,:3]*255.0).astype(np.uint8)
        img_rendered = paint_kpts(None, img_rendered, self.mano_model.kpts_2d_glob[0].cpu().data.numpy())
        return img_rendered
        
    def get_mano_info(self):
        return self.mano_model.get_mano_numpy()
        
    def get_mano_render_info(self):
        return self.mano_model.scale, self.mano_model.xyz_root, self.mano_model.pose_adjusted_all, \
            self.mano_model.shape_adjusted
    
    def get_hand_verts(self):
        self.mano_model.forward_basic()
        return self.mano_model.verts
    
    def reset_parameters(self, keep_mano = False):
        self.mano_model.reset_parameters(keep_mano = keep_mano)
        if not keep_mano:
            self.reset_optimizer_all_state()
        
    def reset_optimizer_all_state(self):
        self.optimizer_adam_mano_fit_all = None
        self.lr_scheduler_all = None