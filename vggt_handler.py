import torch
from torchvision import transforms
from vggt.models.vggt import VGGT
from vggt.heads.dpt_head import DPTHead
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map, closed_form_inverse_se3

import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R
from PIL import Image

import os
import imageio

class VGGTModel:
    def __init__(self, device=None, extract_feats=True, return_images=True, return_depth_conf=True):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        self.dtype = torch.bfloat16
        self.extract_feats = extract_feats
        self.return_images = return_images
        self.return_depth_conf = return_depth_conf
        self.model = self._load_model()
        self.feature_extractor = None
        if self.extract_feats:
            self.feature_extractor = DPTHead(
                dim_in=2*1024,
                patch_size=14,
                features=128,
                feature_only=True,  # Only output features, no activation
                down_ratio=2,  # Reduce spatial memory when feature extraction is enabled.
                pos_embed=False,
            ).eval().to(self.device)
    
    def run(self, batch, depth_batch=None, intrinsics=None, prev_pose_world=None, conf_threshold=0.0, depth_scale=None):
        images = self._preprocess_images(batch) 
        predictions = self._forward(images, conf_threshold)

        T_world_local = self._align_poses(predictions["extrinsic"], prev_pose_world)
        extrinsic = predictions["extrinsic"][-1]
        aligned_pose = T_world_local @ extrinsic

        # pcd = self._generate_point_cloud(predictions, T_world_local, conf_threshold)
        # predictions["pcd"] = pcd 
        predictions["T_world_local"] = T_world_local
        predictions["aligned_pose"] = aligned_pose
        
        # add raw images (single numpy ndarray) into predictions
        images_raw_list = [Image.open(img_path).convert("RGB") for img_path in batch]
        images_raw = np.stack([np.array(img) for img in images_raw_list], axis=0)  # (B, H, W, C)
        predictions["images_raw"] = images_raw
        
        return predictions

    def run_pair(self, batch, conf_threshold):
        images = self._preprocess_images(batch)
        predictions = self._forward(images, conf_threshold)

        extrinsics = predictions["extrinsic"]
        return extrinsics[1]
    
    def save(self, predictions, count, save_images=True, save_pose=True, save_clouds=True): 

        if save_images:
            color = (predictions["images"][2] * 255).clip(0, 255).astype(np.uint8).transpose(1, 2, 0)
            color = np.ascontiguousarray(color)
            color_path = os.path.join("output/images", f"image_{count}.png")
            imageio.imwrite(color_path, color)
        
        if save_pose:
            pose_path = os.path.join("output/poses", f"pose_{count}.txt")
            np.savetxt(pose_path,
                        predictions["T_world_local"] @ predictions["extrinsic"][2], 
                        fmt="%.6f")
            
        if save_clouds:
            o3d.io.write_point_cloud(f"output/clouds/pcd_{count}.pcd", predictions["pcd"])

            

    def _load_model(self):
        print("Loading VGGT")
        return VGGT.from_pretrained("facebook/VGGT-1B").to(self.device)

    def _preprocess_images(self, image_batch):
        return load_and_preprocess_images(image_batch).to(self.device)

    def _forward(self, image_batch, conf_threshold):
        predictions = {}
        with torch.inference_mode():
            with torch.amp.autocast(self.device, dtype=self.dtype):
                images = image_batch[None]
                aggregated_tokens_list, ps_idx = self.model.aggregator(images)
                pose_enc = self.model.camera_head(aggregated_tokens_list)[-1]
                depth_map, depth_conf = self.model.depth_head(aggregated_tokens_list, images, ps_idx)
                feats = None
                if self.extract_feats and self.feature_extractor is not None:
                    feats = self.feature_extractor(aggregated_tokens_list, images, ps_idx)

        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
        extrinsic = extrinsic.squeeze(0)
        extrinsic_inverted = []
        for E in extrinsic:
            E_np = E.detach().to(torch.float32).cpu().numpy()
            E_4x4 = self._to_homogeneous(E_np)  
            # E_inv = np.linalg.inv(E_4x4)[:3, :] 
            E_inv = np.linalg.inv(E_4x4)
            extrinsic_inverted.append(E_inv)

        predictions["extrinsic"] = np.array(extrinsic_inverted)
        predictions["intrinsic"] = intrinsic
        predictions["depth"] = depth_map
        if self.return_depth_conf:
            predictions["depth_conf"] = depth_conf

        # point_map, point_conf = self.model.point_head(aggregated_tokens_list, images, ps_idx)
        # predictions["point_map"] = point_map 
        # predictions["point_conf"] = point_conf


        # print("point map shape: ", point_map.shape)

        if self.return_images:
            predictions["images"] = images
        if feats is not None:
            predictions["feats"] = feats

        for key in predictions.keys():
            if isinstance(predictions[key], torch.Tensor):
                tensor = predictions[key].detach()
                if tensor.dtype == torch.bfloat16:
                    tensor = tensor.to(torch.float32)
                predictions[key] = tensor.cpu().numpy().squeeze(0)  # remove batch dimension and convert to numpy
        
        if self.return_images:
            predictions["images"] = (predictions["images"].transpose(0, 2, 3, 1) * 255).clip(0, 255).astype(np.uint8) # B, H, W, C
        if self.return_depth_conf:
            predictions["depth"] = np.where(predictions["depth_conf"] < conf_threshold, 0.0, predictions["depth"].squeeze(3)) # B, H, W
        else:
            predictions["depth"] = predictions["depth"].squeeze(3)

        for key in predictions.keys():
            if isinstance(predictions[key], np.ndarray):
                predictions[key] = np.ascontiguousarray(predictions[key])
        
        # print(predictions["intrinsic"].shape)
        # print(predictions["extrinsic"].shape)
        # print(predictions["depth"].shape)
        # print(predictions["depth_conf"].shape)
        # print(predictions["images"].shape)

        return predictions 
    
    def _to_homogeneous(self, T_3x4):
        T = np.eye(4)
        T[:3, :] = T_3x4
        return T
    
    def _to_3x4(self, T_4x4):
        return T_4x4[:3, :]
    
    def _align_poses(self, extrinsics, T_world_curr): 
        # T_local_curr = self._to_homogeneous(extrinsics[0])
        T_local_curr = extrinsics[0]

        if T_world_curr is None:
            T_world_local = np.eye(4)
        else:
            T_world_local = T_world_curr @ np.linalg.inv(T_local_curr)
        return T_world_local


    def _generate_point_cloud(self, predictions, T_world_local, conf_threshold):

        depth_map = predictions["depth"]
        depth_conf = predictions["depth_conf"]
        intrinsics = predictions["intrinsic"]
        extrinsics = predictions["extrinsic"]
        images = predictions["images"]

        if images.ndim == 3:  # Single image case
            images = images[None]
            depth_map = depth_map[None]
            depth_conf = depth_conf[None]
            extrinsics = extrinsics[None]
            intrinsics = intrinsics[None]

        pcd_total = o3d.geometry.PointCloud()

        for i in range(len(images)):
            color = (images[i] * 255).clip(0, 255).astype(np.uint8).transpose(1, 2, 0)
            color = np.ascontiguousarray(color)

            conf = depth_conf[i]
            depth_raw = depth_map[i].squeeze(2) 
            depth_filtered = np.where(conf >= conf_threshold, depth_raw, 0.0).astype(np.float32)
            depth = np.ascontiguousarray(depth_filtered)

            extr = extrinsics[i]
            intr = intrinsics[i]

            # print(intr)

            color_o3d = o3d.geometry.Image(color)
            depth_o3d = o3d.geometry.Image(depth)
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_o3d, depth_o3d,
                convert_rgb_to_intensity=False,
                depth_scale=1.0,
                depth_trunc=10.0
            )

            height, width = depth.shape
            intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2])
            # intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, 257, 256, 259, 147)
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
            
            pcd_total+=pcd.transform(extr)
            # pcd_total += pcd

        return pcd_total.transform(T_world_local)
        
        
