import numpy as np
import math

class kpts_global_projector:
    def __init__(self, dataset_name): 
        if dataset_name == 'mano_hand':
            # Order of coordinates in position matrices
            self.X = 1
            self.Y = 0
            self.Z = 2

            # Set dataset parameters
            self.key_bone_l = 10.0           # key bone length in cm
            self.foc_l = 1.0                 # focal length
            self.cam_h = 400                 # pixels
            self.cam_w = 400                 # pixels
            self.PXCM = 357.00               # pixels per cm ratio
            self.wrist_idx = 0               # index for the wrist joint
            self.mmcp_idx = 9                # index for the first joint of middle finger
            self.pmcp_idx = 17               # index for the first joint of pinky finger
        else:
            print("Error. Unsupported dataset {}".format(dataset_name))
            
    def canon_to_global(self, pos_2d, pos_can_3d):
        assert pos_2d.shape == (21, 2)
        assert pos_can_3d.shape == (21, 3)
        pos_2d = pos_2d.copy()
        pos_can_3d = pos_can_3d.copy()
        if np.all(pos_2d == 0) or np.all(pos_can_3d == 0):
            pose_3d_glob = np.zeros(pos_can_3d.shape)
        else:
            pose_3d_glob = self.calculate_global_positions(pos_2d, pos_can_3d)

        return pose_3d_glob.astype(np.float32)

    # Recreates the global positions of a single hand hand, using the 2D image coordinates and 
    # the canonical orientation. hand_2d and hand_3d should have the positions of a single hand
    def calculate_global_positions(self, hand_2d, hand_3d):
        wrist_2d = hand_2d[self.wrist_idx]
        middle_finger_2d = hand_2d[self.mmcp_idx] # Key joint is usually the 1st joint of the middle finger
        wrist_3d = hand_3d[self.wrist_idx]

        # The x and y angles are "swapped" because a rotation along the x axis changes y and z, not x
        # Since the y angle has to be recalculated after the first x rotation to the origin, there are two y angles:
        # post_rot_angle_y is the same as the angle between the middle finger and the z-axis in the xz plane
        # this angle is the angle AFTER rotation around the x-axis, not the angle of rotation around the y-axis
        # angle_y is the actual angle of y-axis rotation, which must be calculated from post_rot_angle_y
        # These angles (angle_x and post_rot_angle_y) are found from the 2D projection of the middle finger's 1st joint
        midfinger_x_2d = ((middle_finger_2d[self.X] * self.cam_w) - self.cam_w/2) / self.PXCM
        post_rot_angle_y = math.atan2(midfinger_x_2d, self.foc_l)
        midfinger_y_2d = ((middle_finger_2d[self.Y] * self.cam_h) - self.cam_h/2) / self.PXCM
        angle_x = math.atan2(midfinger_y_2d, self.foc_l)
        angle_y = math.atan2(midfinger_x_2d, (midfinger_y_2d * math.sin(angle_x) + (midfinger_x_2d * \
            math.cos(angle_x))/(math.tan(post_rot_angle_y))))

        # Use either wrist or PF1 to MF1 as the edge used to calculate 3d global positions (whichever edge is 
        # longer in the image) This prevents an issue where values calculated with 2d distances were off because 
        # the points were too close together to get an accurate measurement
        pinky_2d = hand_2d[self.pmcp_idx]

        dist_wm = math.sqrt((self.cam_w*(wrist_2d[self.X]-middle_finger_2d[self.X]))**2 + \
            (self.cam_h*(wrist_2d[self.Y]-middle_finger_2d[self.Y]))**2)
        dist_pm = math.sqrt((self.cam_w*(pinky_2d[self.X]-middle_finger_2d[self.X]))**2 + \
            (self.cam_h*(pinky_2d[self.Y]-middle_finger_2d[self.Y]))**2)

        # For the partner joint, we choose either the wrist or the pinky (even though MF1->PF1 isn't a bone)
        if(dist_wm > dist_pm):
            joint_2d = wrist_2d
            joint_3d = wrist_3d
        else:
            joint_2d = pinky_2d
            joint_3d = hand_3d[self.pmcp_idx]

        # Calculate the radius of the middle finger to the camera in the 3d global coordinates, 
        # which is the translation distance zt
        zt = self.calc_radius(joint_3d, joint_2d, angle_x, angle_y)

        # Reverse the creation of the canonical matrix to turn the canonical matrix into the global matrix
        # rescale the hand
        hand_3d *= self.key_bone_l
        # translate in the z direction
        hand_3d[:,self.Z] +=  zt 

        # rotate the hand in the oposite direction of the canonical's rotation to (0,0,z)    
        affine_3d = np.transpose(hand_3d)
        angle_x = -angle_x
        angle_y = -angle_y

        rot_x = np.array([[math.cos(angle_x),0,-math.sin(angle_x)],[0,1,0],[math.sin(angle_x),0,math.cos(angle_x)]])
        rot_y = np.array([[1,0,0],[0,math.cos(angle_y),-math.sin(angle_y)],[0,math.sin(angle_y),math.cos(angle_y)]])
        rot = np.dot(rot_x, rot_y)

        base_change = np.dot(rot, affine_3d)
        hand_global_3d = np.transpose(base_change)

        return hand_global_3d
        
    def calc_radius(self, joint_canon, j_2d, angle_x, angle_y):
        # reverse angles of rotation
        angle_x *= -1
        angle_y *= -1
        # rotate middle finger and partner joint, just as the global was rotated to become the canonical
        foc = self.foc_l
        rot_x = np.array([[math.cos(angle_x),0,-math.sin(angle_x)],[0,1,0],[math.sin(angle_x),0,math.cos(angle_x)]])
        rot_y = np.array([[1,0,0],[0,math.cos(angle_y),-math.sin(angle_y)],[0,math.sin(angle_y),math.cos(angle_y)]])
        # The 2d points have foc as their z position
        j_2d = np.array([((j_2d[0] * self.cam_h) - self.cam_h/2) / self.PXCM, ((j_2d[1] * self.cam_w) - \
            self.cam_w/2) / self.PXCM, foc])
        # After this rotation, mf_2d should be [0, 0, rotated_foc]
        j_2d = np.dot(j_2d,rot_x)
        j_2d = np.dot(j_2d,rot_y)
        # Get the sides of the triangle formed by the canonical bone, in the XY / Z plane
        h = self.key_bone_l * (joint_canon[self.X]**2 + joint_canon[self.Y]**2)**.5
        w = self.key_bone_l * joint_canon[self.Z]
        bone_len_2d = (j_2d[0]**2 + j_2d[1]**2)**.5
        foc = j_2d[2]
        # Get r, which is the z distance to the partner joint's canonical point
        r = h * foc / bone_len_2d
        # R is the radius to the middle finger (also the z portion of the scaled and translated middle finger)
        R = r-w
        return R
        
