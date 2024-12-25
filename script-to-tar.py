from pathlib import Path
import torch
import argparse
import os
import cv2
import numpy as np
import copy
import pickle
import ipdb
import shutil
import yaml

from hmr2.configs import CACHE_DIR_4DHUMANS
from hmr2.models import HMR2, SMPL, download_models as download_models_hmr2, load_hmr2, DEFAULT_CHECKPOINT as DEFAULT_CHECKPOINT_HMR2
from hamer.configs import CACHE_DIR_HAMER
from hamer.models import HAMER, download_models  as download_models_hamer, load_hamer, DEFAULT_CHECKPOINT as DEFAULT_CHECKPOINT_HAMER

from hmr2.utils import recursive_to
from hmr2.utils.render_openpose import render_body_keypoints, render_hand_keypoints
from hmr2.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hmr2.utils.renderer import Renderer, cam_crop_to_full
from hamer.datasets.vitdet_dataset import ViTDetDataset as ViTDetDatasetHAMER

from vitpose_model import ViTPoseModel
from typing import List, Dict, Tuple



LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)

from scipy.spatial.transform import Rotation as R
def rotmat_to_aa(rot_matrices):
    """
    Convert a batch of rotation matrices to axis-angle format using scipy.
    
    Parameters:
    rot_matrices: numpy array of shape (batch_size, 3, 3)
                  Batch of rotation matrices
    
    Returns:
    axis_angles: numpy array of shape (batch_size, 3)
                 Batch of axis-angle vectors
    """
    # Create a Rotation object from the rotation matrices
    rotation_obj = R.from_matrix(rot_matrices)
    
    # Convert to axis-angle (rotation vector) representation
    axis_angles = rotation_obj.as_rotvec()  # This gives the axis-angle representation

    return axis_angles
    
def aa_to_rotmat(axis_angles):
    """
    Convert a batch of axis-angle vectors to rotation matrices using scipy.
    
    Parameters:
    axis_angles: numpy array of shape (batch_size, 3)
                 Batch of axis-angle vectors (rotation vector format)
    
    Returns:
    rot_matrices: numpy array of shape (batch_size, 3, 3)
                  Batch of rotation matrices
    """
    # Create a Rotation object from the axis-angle vectors
    rotation_obj = R.from_rotvec(axis_angles)
    
    # Convert to rotation matrices
    rot_matrices = rotation_obj.as_matrix()  # This gives the rotation matrices

    return rot_matrices
    
# to flip the lr params
from typing import Dict, Tuple
def fliplr_params(mano_params: Dict, has_mano_params: Dict) -> Tuple[Dict, Dict]:
    """
    Flip MANO parameters when flipping the image.
    Args:
        mano_params (Dict): MANO parameter annotations.
        has_mano_params (Dict): Whether MANO annotations are valid.
    Returns:
        Dict, Dict: Flipped MANO parameters and valid flags.
    """
    global_orient = mano_params['global_orient'].copy()
    hand_pose = mano_params['hand_pose'].copy()
    betas = mano_params['betas'].copy()
    has_global_orient = has_mano_params['global_orient'].copy()
    has_hand_pose = has_mano_params['hand_pose'].copy()
    has_betas = has_mano_params['betas'].copy()

    global_orient[1::3] *= -1
    global_orient[2::3] *= -1
    hand_pose[1::3] *= -1
    hand_pose[2::3] *= -1

    mano_params = {'global_orient': global_orient.astype(np.float32),
                   'hand_pose': hand_pose.astype(np.float32),
                   'betas': betas.astype(np.float32)
                  }

    has_mano_params = {'global_orient': has_global_orient,
                       'hand_pose': has_hand_pose,
                       'betas': has_betas
                      }

    return mano_params#, has_mano_params

def fliplr_keypoints(joints: np.array, width: float, flip_permutation: List[int]) -> np.array:
    """
    Flip 2D or 3D keypoints.
    Args:
        joints (np.array): Array of shape (N, 3) or (N, 4) containing 2D or 3D keypoint locations and confidence.
        flip_permutation (List): Permutation to apply after flipping.
    Returns:
        np.array: Flipped 2D or 3D keypoints with shape (N, 3) or (N, 4) respectively.
    """
    joints = joints.copy()
    # Flip horizontal
    joints[:, 0] = width - joints[:, 0] - 1
    joints = joints[flip_permutation, :]

    return joints    

def main():

    with open('hmr2/configs_hydra/experiment/default.yaml', 'r') as file:
        cfg = yaml.safe_load(file)
    
    cfg = {k.lower(): v for k,v in cfg['SMPLH'].items()}
    cfg['data_dir'] = os.path.expandvars(cfg['data_dir'])
    os.environ["SMPLH.DATA_DIR"] = cfg['data_dir']
    cfg['joint_regressor_extra'] = os.path.expandvars(cfg['joint_regressor_extra'])
    cfg['mean_params'] = os.path.expandvars(cfg['mean_params'])
    cfg['model_path'] = os.path.expandvars(cfg['model_path'])

    # smplh = SMPL(**cfg)

    import time
    start = time.time()
    parser = argparse.ArgumentParser(description='HMR2 demo code')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT_HMR2, help='Path to pretrained model checkpoint')
    parser.add_argument('--img_folder', type=str, default='example_data/images', help='Folder with input images')
    parser.add_argument('--out_folder', type=str, default='demo_out', help='Output folder to save rendered results')
    parser.add_argument('--side_view', dest='side_view', action='store_true', default=False, help='If set, render side view also')
    parser.add_argument('--top_view', dest='top_view', action='store_true', default=False, help='If set, render top view also')
    parser.add_argument('--full_frame', dest='full_frame', action='store_true', default=False, help='If set, render all people together also')
    parser.add_argument('--save_mesh', dest='save_mesh', action='store_true', default=False, help='If set, save meshes to disk also')
    parser.add_argument('--detector', type=str, default='vitdet', choices=['vitdet', 'regnety'], help='Using regnety improves runtime')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference/fitting')
    parser.add_argument('--file_type', nargs='+', default=['*.jpg', '*.png'], help='List of file extensions to consider')

    args = parser.parse_args()

    # Download and load checkpoints
    # download_models_hmr2(CACHE_DIR_4DHUMANS)
    # model, model_cfg = load_hmr2(args.checkpoint)

    # Setup HMR2.0 model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # model = model.to(device)
    # model.eval()

    # Download and load checkpoints
    download_models_hamer(CACHE_DIR_HAMER)
    model_hamer, model_cfg_hamer = load_hamer(DEFAULT_CHECKPOINT_HAMER)

    # Setup HaMeR model
    model_hamer = model_hamer.to(device)
    model_hamer.eval()    

    # Load detector
    from hmr2.utils.utils_detectron2 import DefaultPredictor_Lazy
    if args.detector == 'vitdet':
        from detectron2.config import LazyConfig
        import hmr2
        cfg_path = Path(hmr2.__file__).parent/'configs'/'cascade_mask_rcnn_vitdet_h_75ep.py'
        detectron2_cfg = LazyConfig.load(str(cfg_path))
        detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        detector = DefaultPredictor_Lazy(detectron2_cfg)
    elif args.detector == 'regnety':
        from detectron2 import model_zoo
        from detectron2.config import get_cfg
        detectron2_cfg = model_zoo.get_config('new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py', trained=True)
        detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
        detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh   = 0.4
        detector       = DefaultPredictor_Lazy(detectron2_cfg)
    
    cpm = ViTPoseModel(device)    

    # Setup the renderer
    # renderer = Renderer(model_cfg, faces=model.smpl.faces)

    # Make output directory if it does not exist
    os.makedirs(args.out_folder, exist_ok=True)

    # Get all demo images that end with .jpg or .png
    img_paths = [img for end in args.file_type for img in Path(args.img_folder).glob(end)]

    # Iterate over all images in folder
    for img_path in img_paths:
        img_cv2 = cv2.imread(str(img_path))
        img = img_cv2.copy()[:, :, ::-1]

        pred_bboxes=[]
        orig_pyd_path = str(img_path)[:-4] + '.data.pyd'
        with open(orig_pyd_path, 'rb') as pyd_file:
            orig_pyd_data = pickle.load(pyd_file)       

        for person_data in orig_pyd_data:    
            box_width = person_data['scale'][0] * 200.
            box_height = person_data['scale'][1] * 200.
            tl_w, tl_h = person_data['center'][0] - box_width / 2, person_data['center'][1] - box_height / 2
            br_w, br_h = person_data['center'][0] + box_width / 2, person_data['center'][1] + box_height / 2
            pred_bboxes.append([tl_w, tl_h, br_w, br_h])

        
        # Detect human keypoints for each person
        vitposes_out = cpm.predict_pose(
            img,
            [np.concatenate([pred_bboxes, np.ones(len(pred_bboxes))[:, None]], axis=1)],
        )

        bboxes = []
        is_right = []
        id_to_boxes_right = {}

        # Use hands based on hand keypoint detections
        for i in range(len(vitposes_out)):
            vitposes = vitposes_out[i]
            left_hand_keyp = vitposes['keypoints'][-42:-21]
            right_hand_keyp = vitposes['keypoints'][-21:]

            # Rejecting not confident detections
            keyp = left_hand_keyp
            valid = keyp[:,2] > 0.5
            if sum(valid) > 3:
                bbox = [keyp[valid,0].min(), keyp[valid,1].min(), keyp[valid,0].max(), keyp[valid,1].max()]
                bboxes.append(bbox)
                is_right.append(0)
            keyp = right_hand_keyp
            valid = keyp[:,2] > 0.5
            if sum(valid) > 3:
                bbox = [keyp[valid,0].min(), keyp[valid,1].min(), keyp[valid,0].max(), keyp[valid,1].max()]
                bboxes.append(bbox)
                is_right.append(1)

            if len(bboxes) == 0:
                continue

            boxes = np.stack(bboxes)
            right = np.stack(is_right)
            id_to_boxes_right[i + 1] = [boxes, right]
            bboxes = []
            is_right = []


        pyd_data = []

        person_id = 1
        for person_data in orig_pyd_data:

            person_pyd_data = {}
            person_pyd_data['personid'] = person_id

            box_center = person_data["center"]
            box_size = person_data["scale"]
            person_pyd_data['center'] = person_data["center"]
            person_pyd_data['scale'] = person_data["scale"]

            person_pyd_data['keypoints_2d'] = person_data["keypoints_2d"]
            person_pyd_data['keypoints_3d'] = person_data["keypoints_3d"]
            person_pyd_data['global_orient'] = person_data["body_pose"][:3]
            person_pyd_data['body_pose'] = person_data["body_pose"][:-6]
            person_pyd_data['betas'] = person_data["betas"]
            person_pyd_data['has_body_pose'] = person_data["has_body_pose"]
            person_pyd_data['has_betas'] = person_data["has_betas"]
            person_pyd_data['extra_info'] = person_data["extra_info"]
            
            person_pyd_data['has_right_hand_pose'] = np.array(0.0)
            person_pyd_data['right_hand_pose'] = np.zeros((45,), dtype=np.float32)
            person_pyd_data['keypoints_2d_right_hand'] = np.zeros((21, 3))     
            person_pyd_data['center_right_hand'] = np.zeros(2) 
            person_pyd_data['scale_right_hand'] = np.zeros(2)
            
            person_pyd_data['has_left_hand_pose'] = np.array(0.0)
            person_pyd_data['left_hand_pose'] = np.zeros((45,), dtype=np.float32)
            person_pyd_data['keypoints_2d_left_hand'] = np.zeros((21, 3))     
            person_pyd_data['center_left_hand'] = np.zeros(2) 
            person_pyd_data['scale_left_hand'] = np.zeros(2)                     

            # Get filename from path img_path
            img_fn, _ = os.path.splitext(os.path.basename(img_path))

            if person_id in id_to_boxes_right:
                # setup to run hamer on this person's hands
                hands_dataset = ViTDetDatasetHAMER(model_cfg_hamer, img_cv2, id_to_boxes_right[person_id][0], id_to_boxes_right[person_id][1], rescale_factor=2.0)
                hamer_dataloader = torch.utils.data.DataLoader(hands_dataset, batch_size=1, shuffle=False, num_workers=0)
                index_to_right_arr = 0
                for hands_batch in hamer_dataloader:
                    hands_batch = recursive_to(hands_batch, device)
                    with torch.no_grad():
                        out_hamer = model_hamer(hands_batch)
                    if id_to_boxes_right[person_id][1][index_to_right_arr] == 1:
                        person_pyd_data['has_right_hand_pose'] = np.array(1.0)
                        person_pyd_data['right_hand_pose'] = rotmat_to_aa(np.array(out_hamer['pred_mano_params']['hand_pose'].squeeze().cpu())).astype(np.float32).flatten()
                        person_pyd_data['keypoints_2d_right_hand'] = np.array(out_hamer['pred_keypoints_2d'].cpu(), dtype=np.float32).squeeze()

                        person_pyd_data['keypoints_2d_right_hand'][:, :2] = person_pyd_data['keypoints_2d_right_hand'][:, :2] * np.array(hands_batch['box_size'].cpu()) + np.array(hands_batch['box_center'].cpu())

                        person_pyd_data['keypoints_2d_right_hand'] = np.hstack([person_pyd_data['keypoints_2d_right_hand'], np.ones((person_pyd_data['keypoints_2d_right_hand'].shape[0], 1))])
                        person_pyd_data['center_right_hand'] = np.array(hands_batch['box_center'].cpu()).squeeze()
                        person_pyd_data['scale_right_hand'] = np.repeat(np.array(hands_batch['box_size'].cpu()), 2) / 200
                    else:
                        # https://github.com/geopavlakos/hamer/issues/78
                        temp_mano_data  = copy.deepcopy(out_hamer)
                        lhand_mano_data = {}
                        lhand_mano_data["betas"]            = temp_mano_data["pred_mano_params"]["betas"][0].cpu().numpy()           # [2,  10]
                        lhand_mano_data["global_orient"]    = temp_mano_data["pred_mano_params"]["global_orient"][0].cpu().numpy()    # [1,  3, 3]
                        lhand_mano_data["hand_pose"]        = temp_mano_data["pred_mano_params"]["hand_pose"][0].cpu().numpy()       # [15, 3, 3]
                        lhand_mano_data["hand_pose"]        = rotmat_to_aa(lhand_mano_data["hand_pose"])                            # [15, 3]
                        lhand_mano_data["betas"]            = lhand_mano_data["betas"].flatten()
                        lhand_mano_data["global_orient"]    = lhand_mano_data["global_orient"].flatten()
                        lhand_mano_data["hand_pose"]        = lhand_mano_data["hand_pose"].flatten()
                        lhand_mano_data                     = fliplr_params(lhand_mano_data, lhand_mano_data)
                        person_pyd_data['has_left_hand_pose'] = np.array(1.0)
                        person_pyd_data['left_hand_pose'] = lhand_mano_data["hand_pose"]

                        person_pyd_data['keypoints_2d_left_hand'] = np.array(out_hamer['pred_keypoints_2d'].cpu(), dtype=np.float32).squeeze()     
                        # flip the keypoints
                        person_pyd_data['keypoints_2d_left_hand'] = fliplr_keypoints(person_pyd_data['keypoints_2d_left_hand'], 1, np.arange(len(person_pyd_data['keypoints_2d_left_hand'])))           
                        # scale the keypoints to the entire image
                        person_pyd_data['keypoints_2d_left_hand'][:, :2] = person_pyd_data['keypoints_2d_left_hand'][:, :2] * np.array(hands_batch['box_size'].cpu()) + np.array(hands_batch['box_center'].cpu())    
                        # add confidence score
                        person_pyd_data['keypoints_2d_left_hand'] = np.hstack([person_pyd_data['keypoints_2d_left_hand'], np.ones((person_pyd_data['keypoints_2d_left_hand'].shape[0], 1))])    
                        person_pyd_data['center_left_hand'] = np.array(hands_batch['box_center'].cpu()).squeeze()
                        person_pyd_data['scale_left_hand'] = np.repeat(np.array(hands_batch['box_size'].cpu()), 2) / 200

                    index_to_right_arr += 1


                # smplh_params = {}
                # smplh_params['global_orient'] = torch.tensor(aa_to_rotmat(person_pyd_data['global_orient'])).float().to('cpu').unsqueeze(0).unsqueeze(0)
                # smplh_params['body_pose'] = torch.tensor(aa_to_rotmat(person_pyd_data['body_pose'][3:].reshape(-1, 3))).float().to('cpu').unsqueeze(0)
                # smplh_params['betas'] = torch.tensor(person_pyd_data['betas']).float().to('cpu').unsqueeze(0)

                # if person_pyd_data['has_right_hand_pose']:
                #     smplh_params['right_hand_pose'] = torch.tensor(aa_to_rotmat(person_pyd_data['right_hand_pose'].reshape(15, 3))).unsqueeze(0).float()
                
                # if person_pyd_data['has_left_hand_pose']:
                #     smplh_params['left_hand_pose'] = torch.tensor(aa_to_rotmat(person_pyd_data['left_hand_pose'].reshape(15, 3))).unsqueeze(0).float()
                
                # smplh_output = smplh(**{k: v for k,v in smplh_params.items()}, pose2rot=False)
                # verts = smplh_output.vertices.detach().squeeze().cpu().numpy()

                # Save all meshes to disk
                # if args.save_mesh:
                #     tmesh = renderer.vertices_to_trimesh(verts, np.zeros(3), LIGHT_BLUE)
                #     tmesh.export(os.path.join(args.out_folder, f'{img_fn}_{person_id}.obj'))

                #     keypoints_img = render_body_keypoints(img, person_pyd_data['keypoints_2d'])

                #     if person_pyd_data['has_right_hand_pose']:
                #         keypoints_img = render_hand_keypoints(keypoints_img, person_pyd_data['keypoints_2d_right_hand'])

                #     if person_pyd_data['has_left_hand_pose']:
                #         keypoints_img = render_hand_keypoints(keypoints_img, person_pyd_data['keypoints_2d_left_hand'])                        

                #     keypoints_img_path = os.path.join(args.out_folder, f'{img_fn}_{person_id}_kepyoints.jpg')
                #     cv2.imwrite(keypoints_img_path, keypoints_img)
            pyd_data.append(person_pyd_data)
            person_id += 1
        
        dest_dir = 'modified-coco-to-include-hands/subset/tar_script_test/'
        new_pyd_path = dest_dir + img_fn + '.data.pyd'
        os.makedirs(os.path.dirname(new_pyd_path), exist_ok=True)
        with open(new_pyd_path, 'wb') as pyd_file:
            pickle.dump(pyd_data, pyd_file) 
        
        shutil.copy(str(img_path), dest_dir)
        shutil.copy(str(img_path)[:-4] + '.detection.npz', dest_dir)

        end = time.time()
        print(end - start)

if __name__ == '__main__':
    main()