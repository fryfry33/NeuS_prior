import torch
import torch.nn.functional as F
import cv2 as cv
import numpy as np
import os
from glob import glob
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp

def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[0].split(' ')[:16]
        else:
            lines = lines[0].split(' ')[:16]
        P = np.array(lines).astype(np.float32).reshape(4, 4)[:3]

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose

class Dataset:
    def __init__(self, conf):
        super(Dataset, self).__init__()
        print('Load data: Begin')
        self.device = torch.device('cuda')
        self.conf = conf

        self.data_dir = conf.get_string('data_dir')
        self.render_cameras_name = conf.get_string('render_cameras_name')
        self.object_cameras_name = conf.get_string('object_cameras_name')

        self.camera_outside_sphere = conf.get_bool('camera_outside_sphere', default=True)
        self.scale_mat_scale = conf.get_float('scale_mat_scale', default=1.1)

        camera_dict = np.load(os.path.join(self.data_dir, self.render_cameras_name))
        self.camera_dict = camera_dict
        
        # --- 1. CHARGEMENT IMAGES (PNG/JPG) ---
        image_dir = os.path.join(self.data_dir)
        files = sorted(glob(os.path.join(image_dir, '*.png')) + 
                       glob(os.path.join(image_dir, '*.jpg')) + 
                       glob(os.path.join(image_dir, '*.jpeg')))
        
        # Tri numérique pour gérer l'ordre 1, 2, 10 correctement
        try:
            files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        except ValueError:
            files.sort()

        self.images_lis = files
        self.n_images = len(self.images_lis)
        if self.n_images == 0:
            raise ValueError(f"Aucune image trouvée dans {image_dir}")

        # Chargement en mémoire RAM (CPU)
        self.images_np = np.stack([cv.imread(im_name) for im_name in self.images_lis]) / 256.0

        # --- 2. CHARGEMENT MASQUES ---
        mask_dir = os.path.join(self.data_dir, 'mask')
        mask_files = sorted(glob(os.path.join(mask_dir, '*.png')) + 
                            glob(os.path.join(mask_dir, '*.jpg')))

        if len(mask_files) > 0:
            try:
                mask_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
            except:
                mask_files.sort()
            self.masks_np = np.stack([cv.imread(im_name) for im_name in mask_files]) / 256.0
        else:
            print("Info: Aucun masque trouvé, utilisation de masques blancs.")
            self.masks_np = np.ones_like(self.images_np)

        # --- 3. CHARGEMENT MATRICES (CORRECTION ID) ---
        self.world_mats_np = []
        self.scale_mats_np = []

        # Au lieu de faire un range(n), on parcourt les fichiers réels
        print("Mapping Images -> Cameras...")
        for img_path in self.images_lis:
            # Extraction ID : "path/4.jpg" -> "4"
            basename = os.path.basename(img_path)
            file_id_str = os.path.splitext(basename)[0]
            
            # Gestion des zéros non significatifs (04 -> 4) si besoin
            try:
                key_id = str(int(file_id_str))
            except ValueError:
                key_id = file_id_str
            
            world_key = f'world_mat_{key_id}'
            scale_key = f'scale_mat_{key_id}'

            # Sécurité : Vérifie si la clé existe dans le .npz
            if world_key not in camera_dict:
                raise KeyError(f"ERREUR CRITIQUE: L'image {basename} cherche la clé '{world_key}' dans cameras_sphere.npz, mais elle n'existe pas.")

            self.world_mats_np.append(camera_dict[world_key].astype(np.float32))
            self.scale_mats_np.append(camera_dict[scale_key].astype(np.float32))

        self.intrinsics_all = []
        self.pose_all = []

        for scale_mat, world_mat in zip(self.scale_mats_np, self.world_mats_np):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = load_K_Rt_from_P(None, P)
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(pose).float())

        # IMAGES SUR CPU (RAM)
        self.images = torch.from_numpy(self.images_np.astype(np.float32)).cpu()
        self.masks  = torch.from_numpy(self.masks_np.astype(np.float32)).cpu()
        
        # MATRICES SUR GPU (VRAM)
        self.intrinsics_all = torch.stack(self.intrinsics_all).to(self.device)
        self.intrinsics_all_inv = torch.inverse(self.intrinsics_all)
        self.focal = self.intrinsics_all[0][0, 0]
        self.pose_all = torch.stack(self.pose_all).to(self.device)
        
        self.H, self.W = self.images.shape[1], self.images.shape[2]
        self.image_pixels = self.H * self.W

        object_bbox_min = np.array([-1.01, -1.01, -1.01, 1.0])
        object_bbox_max = np.array([ 1.01,  1.01,  1.01, 1.0])
        object_scale_mat = self.scale_mats_np[0]
        object_bbox_min = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_min[:, None]
        object_bbox_max = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_max[:, None]
        self.object_bbox_min = object_bbox_min[:3, 0]
        self.object_bbox_max = object_bbox_max[:3, 0]

        print('Load data: End')

    def gen_rays_at(self, img_idx, resolution_level=1):
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).to(self.device)
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, None, :3, :3], p[:, :, :, None]).squeeze()
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)
        rays_v = torch.matmul(self.pose_all[img_idx, None, None, :3, :3], rays_v[:, :, :, None]).squeeze()
        rays_o = self.pose_all[img_idx, None, None, :3, 3].expand(rays_v.shape)
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def gen_random_rays_at(self, img_idx, batch_size):
        # --- FIX CPU/GPU ---
        
        # 1. Conversion index image -> Entier pur
        if isinstance(img_idx, torch.Tensor):
            img_idx = img_idx.cpu().item()
            
        # 2. Génération indices pixels -> FORCÉE SUR CPU
        # device='cpu' est indispensable ici car NeuS met le tenseur par défaut sur CUDA
        pixels_x = torch.randint(low=0, high=self.W, size=(batch_size,), device='cpu')
        pixels_y = torch.randint(low=0, high=self.H, size=(batch_size,), device='cpu')
        
        # 3. Lecture des données (Tout est sur CPU)
        color = self.images[img_idx][(pixels_y, pixels_x)]
        mask = self.masks[img_idx][(pixels_y, pixels_x)]
        
        # 4. Envoi sur GPU pour les maths
        p_x = pixels_x.to(self.device).float()
        p_y = pixels_y.to(self.device).float()
        
        p = torch.stack([p_x, p_y, torch.ones_like(p_y)], dim=-1)
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, :3, :3], p[:, :, None]).squeeze()
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)
        rays_v = torch.matmul(self.pose_all[img_idx, None, :3, :3], rays_v[:, :, None]).squeeze()
        rays_o = self.pose_all[img_idx, None, :3, 3].expand(rays_v.shape)
        
        return torch.cat([rays_o, rays_v, color.to(self.device), mask[:, :1].to(self.device)], dim=-1)

    def get_resolution(self, img_idx):
        return self.W, self.H
        
    def near_far_from_sphere(self, rays_o, rays_d):
        a = torch.sum(rays_d**2, dim=-1, keepdim=True)
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        mid = 0.5 * (-b) / a
        near = mid - 1.0
        far = mid + 1.0
        return near, far
