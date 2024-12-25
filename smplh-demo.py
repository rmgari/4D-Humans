from pathlib import Path
import torch
import argparse
import os
import cv2
import numpy as np
import copy
import pickle
import ipdb

from hmr2.configs import CACHE_DIR_4DHUMANS
from hmr2.models import HMR2, download_models as download_models_hmr2, load_hmr2, DEFAULT_CHECKPOINT as DEFAULT_CHECKPOINT_HMR2
from hamer.configs import CACHE_DIR_HAMER
from hamer.models import HAMER, download_models  as download_models_hamer, load_hamer, DEFAULT_CHECKPOINT as DEFAULT_CHECKPOINT_HAMER

from hmr2.utils import recursive_to
from hmr2.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hmr2.utils.renderer import Renderer, cam_crop_to_full
from hamer.datasets.vitdet_dataset import ViTDetDataset as ViTDetDatasetHAMER

from vitpose_model import ViTPoseModel

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

def main():
    # inspecting_current_out = np.load('hmr2_training_data/dataset_tars/modified-coco-to-include-hands/subset/COCO_train2014_000000570225.detection.npz')
    # # Extract the individual arrays
    # masks = inspecting_current_out['masks']     # Likely a 3D array (num_objects, height, width)
    # scores = inspecting_current_out['scores']   # Likely a 1D array (num_objects,)
    # classes = inspecting_current_out['classes'] # Likely a 1D array (num_objects,)

    # # Example: Inspect first object
    # print("Mask for first object:", masks[0])  # The binary mask for the first object
    # print("Score for first object:", scores[0])  # The confidence score for the first object
    # print("Class for first object:", classes[0])  # The class label for the first object
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
    download_models_hmr2(CACHE_DIR_4DHUMANS)
    model, model_cfg = load_hmr2(args.checkpoint)

    # Setup HMR2.0 model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()

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
    renderer = Renderer(model_cfg, faces=model.smpl.faces)

    # Make output directory if it does not exist
    os.makedirs(args.out_folder, exist_ok=True)

    # Get all demo images that end with .jpg or .png
    img_paths = [img for end in args.file_type for img in Path(args.img_folder).glob(end)]

    # Iterate over all images in folder
    for img_path in img_paths:
        print("NEW IMAGE")
        print(str(img_path))
        img_cv2 = cv2.imread(str(img_path))
        img = img_cv2.copy()[:, :, ::-1]
        

        # Detect humans in image
        det_out = detector(img_cv2)

        det_instances = det_out['instances']
        valid_idx = (det_instances.pred_classes==0) & (det_instances.scores > 0.5)
        boxes=det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()

        # Run HMR2.0 on all detected humans
        dataset = ViTDetDataset(model_cfg, img_cv2, boxes)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)


        pred_bboxes=det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        pred_scores=det_instances.scores[valid_idx].cpu().numpy()

        # Detect human keypoints for each person
        vitposes_out = cpm.predict_pose(
            img,
            [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
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
            id_to_boxes_right[i] = [boxes, right]
            bboxes = []
            is_right = []

        all_verts = []
        all_cam_t = []
        person_count = 1
        pyd_data = []
        for batch in dataloader:
            person_pyd_data = {}
            print("person", person_count)
            person_count += 1 
            batch = recursive_to(batch, device)
            print(batch.keys())
            with torch.no_grad():
                out = model(batch)
            print(out.keys())
            pred_cam = out['pred_cam']
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            person_pyd_data['center'] = np.array(batch["box_center"].cpu(), dtype=np.float32)
            person_pyd_data['scale'] = np.array(batch["box_size"].cpu(), dtype=np.float32)
            img_size = batch["img_size"].float()
            scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()

            # Render the result
            batch_size = batch['img'].shape[0]
            for n in range(batch_size):
                # Get filename from path img_path
                img_fn, _ = os.path.splitext(os.path.basename(img_path))
                person_id = int(batch['personid'][n])
                person_pyd_data['personid'] = person_id              
                white_img = (torch.ones_like(batch['img'][n]).cpu() - DEFAULT_MEAN[:,None,None]/255) / (DEFAULT_STD[:,None,None]/255)
                input_patch = batch['img'][n].cpu() * (DEFAULT_STD[:,None,None]/255) + (DEFAULT_MEAN[:,None,None]/255)
                input_patch = input_patch.permute(1,2,0).numpy()

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
                            out['pred_smpl_params']['right_hand_pose'] = out_hamer['pred_mano_params']['hand_pose']                           
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
                            lhand_mano_data["hand_pose"]        = lhand_mano_data["hand_pose"].reshape(15, 3)                            
                            lhand_mano_data["hand_pose"]        = aa_to_rotmat(lhand_mano_data["hand_pose"])                # [15, 3, 3]
                            out['pred_smpl_params']['left_hand_pose'] = torch.tensor(lhand_mano_data["hand_pose"]).float().to('cuda')                            
                        index_to_right_arr += 1
                
                # pass right + left hand info back to the batch
                batch['pred_smpl_params'] = out['pred_smpl_params']

                
                # Recompute predicted vertices, with logic from forward step
                with torch.no_grad():
                    out = model(batch)

                if 'betas' in out['pred_smpl_params']:
                    person_pyd_data['has_betas'] = True
                    person_pyd_data['betas'] = np.array(out['pred_smpl_params']['betas'].cpu(), dtype=np.float32)
                else:
                    person_pyd_data['has_betas'] = False
                
                if 'body_pose' in out['pred_smpl_params']:
                    person_pyd_data['has_body_pose'] = True
                    person_pyd_data['body_pose'] = np.array(out['pred_smpl_params']['body_pose'].cpu().flatten(), dtype=np.float32)
                else:
                    person_pyd_data['has_body_pose'] = False

                if 'right_hand_pose' in out['pred_smpl_params']:
                    person_pyd_data['has_right_hand_pose'] = True
                    person_pyd_data['right_hand_pose'] = np.array(out['pred_smpl_params']['right_hand_pose'].cpu().flatten(), dtype=np.float32)
                else:
                    person_pyd_data['has_right_hand_pose'] = False

                if 'left_hand_pose' in out['pred_smpl_params']:
                    person_pyd_data['has_left_hand_pose'] = True
                    person_pyd_data['left_hand_pose'] = np.array(out['pred_smpl_params']['left_hand_pose'].cpu().flatten(), dtype=np.float32)
                else:
                    person_pyd_data['has_left_hand_pose'] = False                                            

                if 'pred_keypoints_2d' in out:
                    person_pyd_data['keypoints_2d'] = np.array(out['pred_keypoints_2d'].cpu(), dtype=np.float32)
                if 'pred_keypoints_3d' in out:
                    person_pyd_data['keypoints_3d'] = np.array(out['pred_keypoints_3d'].cpu(), dtype=np.float32)

                print(person_pyd_data.keys())  

                regression_img = renderer(out['pred_vertices'][n].detach().cpu().numpy(),
                                        out['pred_cam_t'][n].detach().cpu().numpy(),
                                        batch['img'][n],
                                        mesh_base_color=LIGHT_BLUE,
                                        scene_bg_color=(1, 1, 1),
                                        )

                final_img = np.concatenate([input_patch, regression_img], axis=1)

                if args.side_view:
                    side_img = renderer(out['pred_vertices'][n].detach().cpu().numpy(),
                                            out['pred_cam_t'][n].detach().cpu().numpy(),
                                            white_img,
                                            mesh_base_color=LIGHT_BLUE,
                                            scene_bg_color=(1, 1, 1),
                                            side_view=True)
                    final_img = np.concatenate([final_img, side_img], axis=1)

                if args.top_view:
                    top_img = renderer(out['pred_vertices'][n].detach().cpu().numpy(),
                                            out['pred_cam_t'][n].detach().cpu().numpy(),
                                            white_img,
                                            mesh_base_color=LIGHT_BLUE,
                                            scene_bg_color=(1, 1, 1),
                                            top_view=True)
                    final_img = np.concatenate([final_img, top_img], axis=1)

                cv2.imwrite(os.path.join(args.out_folder, f'{img_fn}_{person_id}.png'), 255*final_img[:, :, ::-1])

                # Add all verts and cams to list
                verts = out['pred_vertices'][n].detach().cpu().numpy()
                
                cam_t = pred_cam_t_full[n]
                all_verts.append(verts)
                all_cam_t.append(cam_t)

                # Save all meshes to disk
                if args.save_mesh:
                    camera_translation = cam_t.copy()
                    tmesh = renderer.vertices_to_trimesh(verts, camera_translation, LIGHT_BLUE)
                    tmesh.export(os.path.join(args.out_folder, f'{img_fn}_{person_id}.obj'))
        
        # new_pyd_path = 'hmr2_training_data/dataset_tars/modified-coco-to-include-hands/subset/new_pyds/' + img_fn + '.data.pyd'
        # os.makedirs(os.path.dirname(new_pyd_path), exist_ok=True)
        # with open(new_pyd_path, 'wb') as pyd_file:
        #     pickle.dump(pyd_data, pyd_file) 

        # Render front view
        if args.full_frame and len(all_verts) > 0:
            misc_args = dict(
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
                focal_length=scaled_focal_length,
            )
            cam_view = renderer.render_rgba_multiple(all_verts, cam_t=all_cam_t, render_res=img_size[n], **misc_args)

            # Overlay image
            input_img = img_cv2.astype(np.float32)[:,:,::-1]/255.0
            input_img = np.concatenate([input_img, np.ones_like(input_img[:,:,:1])], axis=2) # Add alpha channel
            input_img_overlay = input_img[:,:,:3] * (1-cam_view[:,:,3:]) + cam_view[:,:,:3] * cam_view[:,:,3:]

            cv2.imwrite(os.path.join(args.out_folder, f'{img_fn}_all.png'), 255*input_img_overlay[:, :, ::-1])

        end = time.time()
        print(end - start)

if __name__ == '__main__':
    main()
