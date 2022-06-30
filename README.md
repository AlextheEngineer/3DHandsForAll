# 3D Hands for All (3D hand pose annotation using a single image)
<img src="imgs/UI_sample.png" width="640">
We introduce a 3D hand pose annotation tool that can annotate using a single RGB image. This tool enables you to provide an arbitrary number of 2D keypoints and automatically optimize the MANO hand pose to fit the provided keypoints. It also allows full control of all joint rotations (with physical constraints) for more refined annotation. In addition, we provide pretrained 2D and 3D models to enable automatic annotation. However, manual refinement might be needed for higher accuracy as the estimated 2D/3D keypoints can be imperfect.


Existing annotation methods rely on multi-view settings or depth cameras. Consequently, the collected color-based hand images are captured with laboratory environments as background. The limitation in quantity and diversity of 3D hand pose data in the community makes it challenging for learning-based models to generalize to scenes in the wild. This tool gives everyone the ability to obtain 3D hand pose data using monocular RGB images. It is used in our paper [Ego2HandsPose: A Dataset for Egocentric Two-hand 3D Global Pose Estimation](https://arxiv.org/abs/2206.04927) for the annotation of the Ego2Hands dataset, which can be used to composite two-hand training images with significantly higher quantity and diversity. Click here for more details on [Ego2Hands](https://github.com/AlextheEngineer/Ego2Hands).

## Environment Setup
* We used the following steps to set up the proper environment in Anaconda on a Windows 10 machine:
  > conda create --name hand3d_env python=3.7\
  > conda activate hand3d_env\
  > conda install -c conda-forge opencv\
  > conda install pytorch torchvision torchaudio cudatoolkit=10.1 -c pytorch (see https://pytorch.org/ for a proper setting specific for your machine)\

To install MANO hand, download the repository from https://github.com/hassony2/manopth and move the root folder into "models/MANO_layer". The "setup.py" file should be located at "models/MANO_layer/manopth/setup.py". To install, navigate to the same directory "setup.py" is located in and run
  > pip install .

To install pytorch3d, please follow the official instructions [here](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md).

## Usage
1. **General Introduction**
<img src="imgs/general_intro.png" width="640">

  * hi
  * hi2
3. **Detailed Usage**
