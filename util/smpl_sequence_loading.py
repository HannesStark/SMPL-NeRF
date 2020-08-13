import smplx
import numpy as np
import sys, os
import torch
from vedo import show, Mesh

def load_pose_sequence(file_path: str, device: str, visualize: str = False) -> torch.Tensor:
    """
    Load pose parameters from an AMASS-sequence.
    (The SMPL model requires an (1, 69)-tensor but the AMASS pose-sequence
     contains only 66 pose parameters --> set the remaining pose parameters
     to zero?)

    Parameters
    ----------
    file_path : str
        path to .npz sequence file.
    device : str
        'cpu' or 'cuda:0'.
    visualize : str, optional
        visualize frames from the sequence with vedo. The default is False.

    Returns
    -------
    pose_sequence : torch.Tensor ([n_frames, 1, 69])
        pose tensor for SMPL model.

    """
    bdata = np.load(file_path)
    n_frames = bdata['poses'].shape[0]
    # print('Data keys available:%s'%list(bdata.keys()))
    # print('Vector poses has %d elements for each of %d frames.'%(bdata['poses'].shape[1], bdata['poses'].shape[0]))
    # print('Vector dmpls has %d elements for each of %d frames.'%(bdata['dmpls'].shape[1], bdata['dmpls'].shape[0]))
    # print('Vector trams has %d elements for each of %d frames.'%(bdata['trans'].shape[1], bdata['trans'].shape[0]))
    # print('Vector betas has %d elements constant for the whole sequence.'%bdata['betas'].shape[0])
    # print('The subject of the mocap sequence is %s.'%bdata['gender'])
    pose_sequence = torch.zeros(n_frames, 69).to(device)
    pose_sequence[..., :63] = torch.Tensor(bdata['poses'][:, 3:66])
    print(pose_sequence[..., :3])
    pose_sequence = pose_sequence.view(-1, 1, 69)
    root_orient = torch.Tensor(bdata['poses'][:, :3])
    print(root_orient.shape)
    if visualize:
        smpl_file_name = "../SMPLs/smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl"
        fId = 0 # frame id of the mocap sequence
        frames = np.arange(0, n_frames, 10)
        for fId in frames:
            root_orient = torch.Tensor(bdata['poses'][fId:fId+1, :3]).to(device) # controls the global root orientation
            # pose_body = torch.Tensor(bdata['poses'][fId:fId+1, 3:66]).to(device) # controls the body
            # pose_hand = torch.Tensor(bdata['poses'][fId:fId+1, 66:]).to(device) # controls the finger articulation
            betas = torch.Tensor(bdata['betas'][:10][np.newaxis]).to(device) # controls the body shape
            # dmpls = torch.Tensor(bdata['dmpls'][fId:fId+1]).to(device) # controls soft tissue dynamics
            model = smplx.create(smpl_file_name, model_type='smpl')
            model = model.to(device)
            output = model(betas=betas,
                       return_verts=True, body_pose=pose_sequence[fId])
            vertices = output.vertices.detach().cpu().numpy().squeeze()
            faces = model.faces
            smpl_mesh = Mesh([vertices, faces], alpha=0.5)
            show([smpl_mesh], at=root_orient)
        
    return pose_sequence, root_orient

if __name__ == "__main__":
    npz_bdata_path = '../SMPLs/ACCAD/ACCAD/Male1Walking_c3d/Walk B17 - Walk 2 hop 2 walk_poses.npz' # the path to body data
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    load_pose_sequence(npz_bdata_path, device, visualize=False)
    