import sys
import numpy as np
import torch
import torch.nn as nn
from manopth.manolayer import ManoLayer
import pytorch3d
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_view_transform, look_at_rotation, 
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, PointLights, TexturesVertex,
)
import utils

def projectPoints(xyz, K):
    """ Project 3D coordinates into image space. """
    #uv = torch.matmul(K, xyz.T).T
    uv = torch.matmul(K, xyz.transpose(2, 1)).transpose(2, 1)
    return uv[:, :, :2] / uv[:, :, -1:]

class Model(nn.Module):
    def __init__(self, mano_path, renderer, device, batch_size, root_idx = 0):
        super().__init__()

        self.device = device
        self.renderer = renderer
        self.wrist_idx = 0
        self.mcp_idx = 9
        self.root_idx = root_idx
        self.key_bone_len = 10.0 #cm
        self.mano_layer = ManoLayer(center_idx = self.root_idx,
            flat_hand_mean=False,
            side="right",
            mano_root=mano_path,
            ncomps=45,
            use_pca=False,
            root_rot_mode="axisang",
            joint_rot_mode="axisang").cuda()
        self.img_center_size = 200
        self.batch_size = batch_size
        self.K = torch.tensor([[357, 0, self.img_center_size],[0, 357, self.img_center_size],[0,0,1]], dtype=torch.float32).cuda()
        self.xyz_root = torch.tensor([0.0, 0.0, 50.0], dtype=torch.float32).cuda().repeat(self.batch_size, 1)
        self.input_pose = torch.zeros(self.batch_size, 48).cuda()
        self.input_shape = torch.zeros(self.batch_size, 10).cuda()
        self.camera_position = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32).cuda().repeat(1, 1)
        self.ups = torch.tensor([0.0,-1.0,1.0], dtype=torch.float32).cuda().repeat(1, 1)
        self.ats = torch.tensor([0.0,0.0,1.0], dtype=torch.float32).cuda().repeat(1, 1)
        self.image_render = None
        
        mano_layer_left = ManoLayer(center_idx = self.root_idx,
            flat_hand_mean=False,
            side="left",
            mano_root=mano_path,
            ncomps=45,
            use_pca=False,
            root_rot_mode="axisang",
            joint_rot_mode="axisang").cuda()
        self.verts0 = None
        self.verts1 = None
        self.faces0 = mano_layer_left.th_faces.repeat(1, 1, 1)
        self.faces1 = self.mano_layer.th_faces.repeat(1, 1, 1)
        self.texture = TexturesVertex(verts_features=torch.ones(1, 778, 3).cuda())
        #self.faces = self.mano_layer.th_faces.repeat(self.batch_size, 1, 1)
        
        self.set_up_camera()

    def set_up_camera(self):
        self.R = look_at_rotation(self.camera_position, at=self.ats, up=self.ups, device=self.device)
        self.T = -torch.bmm(self.R.transpose(1, 2), self.camera_position[:,:,None])[:, :, 0]

    """
    def set_two_hand_pose(self, scale, xyz_root, input_pose, input_shape):
        assert xyz_root.shape == (self.batch_size, 3)
        assert input_pose.shape == (self.batch_size, 48)
        assert input_shape.shape == (self.batch_size, 10)
        self.scale = scale
        self.xyz_root.data = xyz_root.clone().detach()
        self.input_pose.data = input_pose.clone().detach()
        self.input_shape.data = input_shape.clone().detach()
    """
    
    def set_verts(self, verts0, verts1):
        self.verts0 = verts0
        self.verts1 = verts1        

    def forward(self):
        """
        hand_verts, hand_joints = self.mano_layer(self.input_pose, self.input_shape)
        
        # Flip left hand back
        #hand_verts[0,:,0] *= -1.0
        #hand_joints[0,:,0] *= -1.0

        # Shifting & scaling
        #print(hand_joints.shape)
        #print(self.scale.shape)
        hand_joints = hand_joints/(self.scale[:,None,:])
        hand_joints += self.xyz_root[:,None,:]
        # Calculate the 2D loss
        uv_mano_full = projectPoints(hand_joints, self.K)
        uv_mano_full[torch.isnan(uv_mano_full)] = 0.0
        kpts_2d_glob = uv_mano_full.clone().detach()[:, :, [1, 0]]

        verts = hand_verts/(self.scale[:,None,:])
        verts += self.xyz_root[:,None,:]
        
        
        verts_rgb = torch.ones_like(verts[0:1])
        textures0 = TexturesVertex(verts_features=verts_rgb.to(self.device))
        
        #verts_rgb0 = torch.ones_like(verts[0:1])
        #textures0 = TexturesVertex(verts_features=verts_rgb0.cuda())
        
        #verts_rgb1 = torch.ones_like(verts[1:2])
        #textures1 = TexturesVertex(verts_features=verts_rgb1.cuda())
        
        #verts[0] = verts[1]
        verts[0,:,0] *= -1.0
        
        hand_mesh0 = Meshes(
            verts=verts[0:1],
            faces=self.faces[0:1],
            textures=textures0
        )
        
        hand_mesh1 = Meshes(
            verts=verts[1:2],
            faces=self.faces[1:2],
            textures=textures0
        )
        """
        mesh_list = []
        if self.verts0 is not None:
            self.verts0[0,:,0] *= -1.0
            
            hand_mesh0 = Meshes(
                verts=self.verts0,
                faces=self.faces0,
                textures=self.texture
            )
            mesh_list.append(hand_mesh0)
        if self.verts1 is not None:
            hand_mesh1 = Meshes(
                verts=self.verts1,
                faces=self.faces1,
                textures=self.texture
            )
            mesh_list.append(hand_mesh1)
        
        hand_mesh = pytorch3d.structures.meshes.join_meshes_as_scene(mesh_list)

        img_rendered = self.renderer(meshes_world=hand_mesh, R=self.R, T=self.T)
        img_rendered = (img_rendered.cpu().data.numpy()[0][:,:,:4]*255.0).astype(np.uint8)
        #img_rendered = paint_kpts(None, img_rendered, self.mano_model.kpts_2d_glob[0].cpu().data.numpy())
        return img_rendered