import numpy as np
import cv2
import sys
from concurrent.futures import ThreadPoolExecutor
import torch
import kornia as K
import kornia.feature as KF
from utils.logging_utils import Log
import os
from utils.loop_refinement import make_pypose_Sim3, SE3_to_Sim3, pose_refinement
from utils.scale_recovery import find_scale_from_depth
from utils.pose_utils import poses_to_c2w_tensor, convert_pose_numpy_to_opencv_vectorized
from utils.slam_utils import get_matched_camera_points_vectorized
import pypose as pp
from utils.slam_viz import plot_camera_poses, draw_concat_keypoints
from sklearn import linear_model
from lightglue import LightGlue, SuperPoint, DISK, ALIKED, SIFT

class CameraModel(object):
    """
    Class that represents a pin-hole camera model (or projective camera model).
    In the pin-hole camera model, light goes through the camera center (cx, cy) before its projection
    onto the image plane.
    """
    def __init__(self, params):
        """
        Creates a camera model

        Arguments:
            params {dict} -- Camera parameters
        """

        # Image resolution
        self.width = float(params['width'][0])
        self.height = float(params['height'][0])
        # Focal length of camera
        self.fx = float(params['fx'][0])
        self.fy = float(params['fy'][0])
        # Optical center (principal point)
        self.cx = float(params['cx'][0])
        self.cy = float(params['cy'][0])
        
        self.mat = np.array([
                [self.fx, 0, self.cx],
                [0, self.fy, self.cy],
                [0, 0, 1]])

class ClassicTracking:

    def __init__(self, cam_params, loop_enable, viz, config, save_dir = None):

        self.config = config

        self.loop_enable = loop_enable
        self.viz = viz
        self.save_dir = save_dir
        self.retrieval = None

        if self.loop_enable:
            try:
                from utils.loop_closure.retrieval import RetrievalDBOW
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "Loop closure is enabled, but dpretrieval/DBoW2 could not be loaded. "
                    "This usually means the compiled DPRetrieval/DBoW2 binaries are linked "
                    "against a missing OpenCV shared library."
                ) from exc
            self.retrieval = RetrievalDBOW(vocab_path=config["SLAM"].get("vocab_path", "ORBvoc.txt"))
        
        self.cam = CameraModel(cam_params)

        self.K = np.array([
            [cam_params['fx'][0], 0, cam_params['cx'][0]],
            [0, cam_params['fy'][0], cam_params['cy'][0]],
            [0,                   0,                   1]
        ])

        self.prev_frame = None 
        self.prev_rgb = None
        self.feat_ref = None 
        self.feat_curr = None 
        self.detector_method = "FAST"  
        self.min_num_features = 2500
        self.R = np.eye(3)
        self.t = np.zeros((3, 1))
        self.prev_depth = None
        self.pose = np.zeros((4, 4)) 
        self.w2c = np.zeros((4, 4))

        self.mean_depth_array = np.array([])
        self.flag = True
        
        
        self.track_cfg = self.config["tracking"]
        self.device = torch.device(self.track_cfg.get("device", "cuda"))
        
        
        self.feature_mode = self.track_cfg["feature_mode"]
        self.resize_width = self.track_cfg["resize_width"]
        self.extractor_name = self.track_cfg["extractor_name"]
        self.max_keypoints = self.track_cfg["max_keypoints"]
        
        
        self.matching_method = "LightGlue" 
        print(f"[Init] Feature Extractor: {self.extractor_name}")

        # ==================================================
        # Initialize the configured lightweight tracking extractor.
        # ==================================================
        
        
        if self.extractor_name == "DISK":
            print(f"[Init] Loading Tracking Model: DISK (Kornia impl)...")
            self.track_extractor = None 
            self.track_matcher = None
            
        
        else:
            print(f"[Init] Loading Tracking Model: {self.extractor_name} (LightGlue impl)...")
            extractor_map = {
                "SuperPoint": SuperPoint, "ALIKED": ALIKED, "SIFT": SIFT
            }
            if self.extractor_name not in extractor_map:
                raise ValueError(f"Unsupported extractor: {self.extractor_name}")
                
            self.track_extractor = extractor_map[self.extractor_name](
                max_num_keypoints=self.max_keypoints
            ).eval().to(self.device)
            
            self.track_matcher = LightGlue(
                features=self.extractor_name.lower()
            ).eval().to(self.device)
        
        # print(f"[Init] Loading Tracking Model: {self.extractor_name}...")
        
        # extractor_map = {
        #     "SuperPoint": SuperPoint, "DISK": DISK, "ALIKED": ALIKED, "SIFT": SIFT
        # }
        
        # if self.extractor_name not in extractor_map:
        #     raise ValueError(f"Unsupported extractor: {self.extractor_name}")
            
        
        # self.track_extractor = extractor_map[self.extractor_name](
        #     max_num_keypoints=self.track_cfg["max_keypoints"]
        # ).eval().to(self.device)
        
        
        # self.track_matcher = LightGlue(
        #     features=self.extractor_name.lower()
        # ).eval().to(self.device)

        # ==================================================
        # Use a separate DISK extractor with 256 features for loop matching.
        # ==================================================
        
        
        print(f"[Init] Loading Loop-Closure Model: DISK (Fixed)...")
        self.disk_matcher = KF.LightGlueMatcher("disk").eval().to(self.device)
        self.disk_extractor = KF.DISK.from_pretrained("depth").to(self.device)
        
        print(f"[Init] Models Initialized on {self.device}.")
        
        # ==================================================
        # Load MASt3R only when hybrid tracking requires it.
        # ==================================================
        self.heavy_model = None
        need_heavy = self.feature_mode == 'hybrid'
        if need_heavy:
            self._load_mast3r()

        # flow status
        self.prev_kf = None
        self.feat_prev_kf = None

        self.kp_frame_id = 0 # frame id so far

        self.pose = np.eye(4)  # pose matrix [R | t; 0  1]

        self.pose_ref = np.eye(4)

        self.mean_depth_array = np.array([])

        self.kp_ref = None
        self.kp_curr = None


        n_max = config['SLAM']['n_max'] 
        self.pose_history = np.repeat(np.eye(4)[np.newaxis, :, :], n_max, axis=0)  # Initialize
        self.pose_history_old = np.repeat(np.eye(4)[np.newaxis, :, :], n_max, axis=0)  # Initialize

        self.kp_history = []
        self.depth_history = []
        self.frame_history = []
        self.loop_kf_idx = 0

        # Loop Closure Vars
        self.loop_ii = torch.zeros(0, dtype=torch.long)
        self.loop_jj = torch.zeros(0, dtype=torch.long)
        
        
        self.loop_ii = torch.zeros(0, dtype=torch.long)
        self.loop_jj = torch.zeros(0, dtype=torch.long)
        
        from utils.tracking_diagnoser import TrackingDiagnoser
        self.diagnoser = TrackingDiagnoser(self.save_dir, viz_depth_dist=True)

    def _load_mast3r(self):
        from utils.mast3r_matching import MASt3RMatcher
        # An explicitly requested hybrid run must not silently become a single run.
        self.heavy_matcher = MASt3RMatcher(self.track_cfg["heavy_model"], self.device)
        self.heavy_model = self.heavy_matcher.model
        Log("MASt3R loaded in evaluation mode")

    def viz_pose(self, frame_id):
        if not self.viz:
            return
        
        Log(f'Visualing of pose.')
        pred_poses = self.pose_history[:frame_id, :, :]

        out_filename = os.path.join(self.save_dir,'pred_poses.png') if self.save_dir is not None else 'pred_poses.png'
        plot_camera_poses(pred_poses, torch.cat((self.loop_ii, self.loop_jj)), out_filename)

    def get_loop_update(self):
        return self.pose_history_old[:self.kp_frame_id, :, :], self.pose_history[:self.kp_frame_id, :, :]
    # Keep a keyframe reference for inexpensive optical-flow gating.
    def set_curr_keyframe(self, frame):
        
        
        frame_normalized = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
        frame_uint8 = frame_normalized.astype(np.uint8)
        frame = cv2.cvtColor(np.transpose(frame_uint8, (1, 2, 0)), cv2.COLOR_RGB2GRAY)
        
        
        self.prev_kf = frame
        
        
        feat_ref = self.detect_features(frame)
        
        
        self.feat_prev_kf = np.array([x.pt for x in feat_ref], dtype=np.float32)
    # Measure image motion before running feature matching and mapping.
    def motion_flow_keyframe(self, frame, prev_kf = True):
        
        frame_normalized = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
        frame_uint8 = frame_normalized.astype(np.uint8)
        frame = cv2.cvtColor(np.transpose(frame_uint8, (1, 2, 0)), cv2.COLOR_RGB2GRAY)  
        
        if prev_kf:     
            
            kp2, st, err = cv2.calcOpticalFlowPyrLK(self.prev_kf, frame,
                                                    self.feat_prev_kf, None,    
                                                    winSize=(21, 21),
                                                    criteria=(
                                                    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

            st = st.reshape(st.shape[0])  
            
            kp1 = self.feat_prev_kf[st == 1]
            kp2 = kp2[st == 1]
        else:         
            ### 8 ms avg ###
            kp2, st, err = cv2.calcOpticalFlowPyrLK(self.prev_frame, frame,
                                                    self.feat_ref, None,        

                                                    winSize=(21, 21),
                                                    criteria=(
                                                    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

            st = st.reshape(st.shape[0])  # status of points from frame to frame
            # Keypoints
            kp1 = self.feat_ref[st == 1]
            kp2 = kp2[st == 1]
        
        
        flow_vectors = kp2 - kp1
        
        flow_magnitudes = np.linalg.norm(flow_vectors, axis=1)
        
        # Use median flow magnitude to reduce the influence of outliers.
        
        return np.median(flow_magnitudes)

    def detect_features(self, frame):
        """
        Point-feature detector: search for salient keypoints that are likely to match well in other image frames.
            - corner detectors: Moravec, Forstner, Harris, Shi-Tomasi, and FAST.
            - blob detectors: SIFT, SURF, and CENSURE.

        Args:
            frame {ndarray}: frame to be processed
        """

        if self.detector_method == "FAST":
            detector = cv2.FastFeatureDetector_create()  # threshold=25, nonmaxSuppression=True)
            return detector.detect(frame)

        elif self.detector_method == "ORB":
            detector = cv2.ORB_create(nfeatures=3000)
            kp1, des1 = detector.detectAndCompute(frame, None)
            return kp1
    
    def heavy_matching(self, frame_curr, frame_prev, depth=None, depth_threshold=65):
        if self.heavy_model is None:
            raise RuntimeError("MASt3R requested without a loaded model")
        return self.heavy_matcher.match(frame_prev, frame_curr, depth, depth_threshold)

    def _numpy_to_tensor(self, img):
        """
        Convert HWC or CHW images to batched LightGlue tensors.
        """
        
        # Copy the array to avoid unsupported NumPy strides.
        tensor = torch.from_numpy(img.copy()).float() / 255.0
        
        
        if tensor.dim() == 2: 
            
            tensor = tensor.unsqueeze(0).repeat(3, 1, 1)
            
        elif tensor.dim() == 3:
            
            if tensor.shape[2] == 3: 
                tensor = tensor.permute(2, 0, 1)


        return tensor.unsqueeze(0).to(self.device)

    def feature_matching(self, frame, frame_rgb, depth=None, filter=False, lc_frame_id=None, sky_mask=None, depth_threshold=65):
        from lightglue.utils import rbd 

        
        if lc_frame_id is not None:
            prev_frame = self.frame_history[lc_frame_id]
            feat_ref = self.kp_history[lc_frame_id]
        else:
            prev_frame = self.prev_rgb
            feat_ref = self.feat_ref

        kp1, kp2 = None, None

        # ================== 1. Deep Matching ==================
        if self.matching_method == "LightGlue":
            
            
            # Keep the Kornia DISK path for loop closure and DISK tracking.
            use_kornia_disk = (lc_frame_id is not None) or (self.extractor_name == "DISK")

            if use_kornia_disk:


                def ensure_hwc(img):
                    
                    if img.ndim == 3 and img.shape[0] == 3 and img.shape[2] > 10:
                        return img.transpose(1, 2, 0)
                    return img
                
                
                prev_frame_k = cv2.cvtColor(prev_frame, cv2.COLOR_GRAY2RGB) if len(prev_frame.shape)==2 else prev_frame
                frame_k = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB) if len(frame.shape)==2 else frame
                
                prev_frame_k = ensure_hwc(prev_frame_k)
                frame_k = ensure_hwc(frame_k)
                
                
                # print(f"\n[DEBUG] Feature Matching Shapes:")
                # print(f"  Prev Frame (np): {prev_frame_k.shape}")
                # print(f"  Curr Frame (np): {frame_k.shape}")
                
                img1 = K.utils.image_to_tensor(prev_frame_k).unsqueeze(0).float() / 255.0
                img2 = K.utils.image_to_tensor(frame_k).unsqueeze(0).float() / 255.0
                img1 = img1.to(self.device)
                img2 = img2.to(self.device)
                
                hw1 = torch.tensor(img1.shape[2:], device=self.device)
                hw2 = torch.tensor(img2.shape[2:], device=self.device)
                # print("shape of hw",hw1.shape, hw2.shape)
                # print("shape of img",img1.shape, img2.shape)

                
                if lc_frame_id is not None:
                    num_features = 256  
                else:
                    num_features = self.max_keypoints 

                with torch.inference_mode():
                    
                    inp = torch.cat([img1, img2], dim=0)
                    
                    
                    features1, features2 = self.disk_extractor(inp, num_features, pad_if_not_divisible=True)
                    
                    kps1, descs1 = features1.keypoints, features1.descriptors
                    kps2, descs2 = features2.keypoints, features2.descriptors
                    
                    # Build local affine frames for the Kornia LightGlue matcher.
                    lafs1 = KF.laf_from_center_scale_ori(kps1[None], torch.ones(1, len(kps1), 1, 1, device=self.device))
                    lafs2 = KF.laf_from_center_scale_ori(kps2[None], torch.ones(1, len(kps2), 1, 1, device=self.device))
                    
                    
                    dists, idxs = self.disk_matcher(descs1, descs2, lafs1, lafs2, hw1=hw1, hw2=hw2)

                
                kp1 = kps1[idxs[:, 0]].cpu().numpy()
                kp2 = kps2[idxs[:, 1]].cpu().numpy()

            
            else:
                # print(prev_frame.shape, frame_rgb.shape)
                
                img0_tensor = self._numpy_to_tensor(prev_frame)
                img1_tensor = self._numpy_to_tensor(frame_rgb)
                # print(img0_tensor.shape, img1_tensor.shape)
                
                with torch.inference_mode():
                    
                    feats0 = self.track_extractor.extract(img0_tensor, resize=self.resize_width)
                    feats1 = self.track_extractor.extract(img1_tensor, resize=self.resize_width)
                    
                    
                    matches01 = self.track_matcher({'image0': feats0, 'image1': feats1})
                    
                    
                    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]
                    matches = matches01['matches']
                    
                    
                    kp1 = feats0['keypoints'][matches[..., 0]].cpu().numpy()
                    kp2 = feats1['keypoints'][matches[..., 1]].cpu().numpy()

        
        elif self.matching_method == "OF_PyrLK":
            if len(prev_frame.shape) == 3: prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            else: prev_gray = prev_frame
            if len(frame.shape) == 3: curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else: curr_gray = frame

            kp2, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, feat_ref, None,
                                                    winSize=(21, 21),
                                                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
            st = st.reshape(st.shape[0])
            kp1 = feat_ref[st == 1]
            kp2 = kp2[st == 1]

        
        if depth is not None and depth_threshold is not None and kp2 is not None and len(kp2) > 0:
            u = np.round(kp2[:, 0]).astype(int)
            v = np.round(kp2[:, 1]).astype(int)
            u = np.clip(u, 0, depth.shape[1] - 1)
            v = np.clip(v, 0, depth.shape[0] - 1)
            
            matched_depths = depth[v, u]
            if sky_mask is None:
                valid_mask = matched_depths < depth_threshold
            else:
                valid_mask = (matched_depths < depth_threshold) & (sky_mask[v, u] > 0.5)
            
            kp1 = kp1[valid_mask]
            kp2 = kp2[valid_mask]

        
        if filter and kp1 is not None and len(kp1) > 0:
            threshold = 15.0
            # print(kp1.shape, kp2.shape)
            # print(kp1[:5], kp2[:5])
            # print(np.linalg.norm(kp2 - kp1, axis=1))
            movement = np.sqrt(np.linalg.norm(kp2 - kp1, axis=1))

            valid_indices = movement < threshold
            kp1 = kp1[valid_indices]
            kp2 = kp2[valid_indices]

        return kp1, kp2

    def motion_estimation_init(self, frame):
        """
        Processes first frame to initialize the reference features and the matrix R and t.
        Only for frame_id == 0.

        Args:
            frame {ndarray}: frame to be processed
            frame_id {int}: integer corresponding to the frame idself.loop_kf_idx
        """
        feat_ref = self.detect_features(frame)
        self.feat_ref = np.array([x.pt for x in feat_ref], dtype=np.float32)

    def loop_kf(self, frame, frame_id):
        if self.loop_enable:
            frame = frame.transpose(1, 2, 0) 
            frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)   # Downsample before ORB retrieval to reduce its cost.
            self.retrieval(frame, frame_id)     
        else:
            Log('Loop Closure is disabled!')

    
    def calculate_scale_correction(self, last_img, last_depth, depth_uni_curr, R, frame_id_for_debug=0,sky_mask = None):
        debug_data = {
            "render_rgb": None, "render_depth": None, "mask_trust": None,
            "scatter_x": None, "scatter_y": None, "ransac_line": None,
            "status": "Init", "feat_curr": None
        }

        if last_img is None or last_depth is None:
            return 1.0, debug_data

        if isinstance(depth_uni_curr, torch.Tensor):
            depth_uni_curr = depth_uni_curr.detach().cpu().numpy()
        # depth_uni_curr = np.squeeze(depth_uni_curr)
        
        # -----------------------------------------------------------
        
        # -----------------------------------------------------------
        gt_rgb = self.prev_rgb # CPU numpy array
        
        
        if gt_rgb.ndim == 3 and gt_rgb.shape[0] == 3 and gt_rgb.shape[2] != 3:
            gt_rgb = np.transpose(gt_rgb, (1, 2, 0))
        
        
        tgt_h, tgt_w = gt_rgb.shape[:2]

        # -----------------------------------------------------------
        
        # -----------------------------------------------------------
        render_rgb = last_img.detach().cpu().numpy().squeeze()
        
        # CHW -> HWC
        if render_rgb.ndim == 3 and render_rgb.shape[0] == 3:
            render_rgb = np.transpose(render_rgb, (1, 2, 0))
        
        
        if render_rgb.shape[:2] != (tgt_h, tgt_w):
            render_rgb = cv2.resize(render_rgb, (tgt_w, tgt_h), interpolation=cv2.INTER_LINEAR)

        
        if render_rgb.max() <= 1.1:
            render_rgb = render_rgb * 255.0

        debug_data["render_rgb"] = render_rgb.astype(np.uint8)

        # -----------------------------------------------------------
        
        # -----------------------------------------------------------
        render_depth = last_depth.detach().cpu().numpy().squeeze()
        if render_depth.ndim == 3: render_depth = render_depth[0]
        
        if render_depth.shape[:2] != (tgt_h, tgt_w):
            render_depth = cv2.resize(render_depth, (tgt_w, tgt_h), interpolation=cv2.INTER_NEAREST)
            
        debug_data["render_depth"] = render_depth

        # -----------------------------------------------------------
        # Select regions with photometric and geometric agreement.
        # -----------------------------------------------------------
        
        gt_rgb_f = gt_rgb.astype(np.float32)
        render_rgb_f = render_rgb.astype(np.float32)
        diff = np.abs(gt_rgb_f - render_rgb_f)
        diff_mean = np.mean(diff, axis=2)
        mask_rec = diff_mean < 30.0 

        
        if sky_mask is None:
            binary_sky_mask = (render_depth < 100.0) & (depth_uni_curr < 100.0)
        else:
            binary_sky_mask = sky_mask.astype(bool)
        
        
        mask_geo = (render_depth > 1) & (depth_uni_curr > 1)
        
        
        # mask_geo = (render_depth > 3) & (render_depth < 100.0)
        
        # Exclude distant geometry beyond the trusted depth range.
        # mask_range = (depth_uni_curr > 3) & (depth_uni_curr < 100.0)

        # Reject inconsistent predicted/rendered depth pairs.
        # mask_logic = ~(((depth_uni_curr > 50.0) & (render_depth < 10.0)) | ((depth_uni_curr < 10.0) & (render_depth > 50.0)))

        
        # mask_trust = mask_rec & mask_geo & mask_range & mask_logic
        # mask_trust = mask_rec & mask_geo & mask_logic
        mask_trust = mask_rec & mask_geo & binary_sky_mask
        debug_data["mask_trust"] = mask_trust
        valid_cnt = np.sum(mask_trust)
        if valid_cnt < 500: 
            debug_data["status"] = "Mask Empty"
            return 1.0, debug_data

        # -----------------------------------------------------------
        
        # -----------------------------------------------------------
        if hasattr(self, 'feat_curr') and self.feat_curr is not None:
            feats_c = self.feat_curr
            
            if isinstance(feats_c, torch.Tensor):
                feats_c = feats_c.detach().cpu().numpy()
            
            
            debug_data["feat_curr"] = feats_c
        
        # -----------------------------------------------------------
        
        # -----------------------------------------------------------
        kp_ref = self.feat_ref
        u = np.round(kp_ref[:, 0]).astype(int)
        v = np.round(kp_ref[:, 1]).astype(int)
        
        
        bound_mask = (u >= 0) & (u < tgt_w) & (v >= 0) & (v < tgt_h)
        valid_u = u[bound_mask]
        valid_v = v[bound_mask]
        
        
        in_mask = mask_trust[valid_v, valid_u]
        final_indices = np.where(bound_mask)[0][in_mask]
        
        if len(final_indices) < 20:
            # print(f"[ScaleDebug {frame_id_for_debug}] Few Features: {len(final_indices)}")
            debug_data["status"] = "Few Features"
            return 1.0, debug_data

        # -----------------------------------------------------------
        # Project map points into the current camera.
        # -----------------------------------------------------------
        sel_kp_ref = self.feat_ref[final_indices]
        sel_kp_curr = self.feat_curr[final_indices]
        z_ref = render_depth[valid_v[in_mask], valid_u[in_mask]]

        cx, cy, fx, fy = self.cam.cx, self.cam.cy, self.cam.fx, self.cam.fy
        X_ref = (sel_kp_ref[:, 0] - cx) * z_ref / fx
        Y_ref = (sel_kp_ref[:, 1] - cy) * z_ref / fy
        P_ref = np.stack([X_ref, Y_ref, z_ref], axis=1)

        P_curr_rot = P_ref @ R 
        Z_target = P_curr_rot[:, 2]

        pos_z = Z_target > 0.1
        if np.sum(pos_z) < 20: 
            debug_data["status"] = "Z<=0"
            return 1.0, debug_data
        
        Z_target = Z_target[pos_z]
        sel_kp_curr = sel_kp_curr[pos_z]

        # -----------------------------------------------------------
        
        # -----------------------------------------------------------
        u_c = np.round(sel_kp_curr[:, 0]).astype(int)
        v_c = np.round(sel_kp_curr[:, 1]).astype(int)
        h_c, w_c = depth_uni_curr.shape
        valid_c = (u_c >= 0) & (u_c < w_c) & (v_c >= 0) & (v_c < h_c)
        
        if np.sum(valid_c) < 20: return 1.0, debug_data
        
        Z_uni = depth_uni_curr[v_c[valid_c], u_c[valid_c]]
        Z_target = Z_target[valid_c]
        
        debug_data["scatter_x"] = Z_uni
        debug_data["scatter_y"] = Z_target

        # -----------------------------------------------------------
        # Compare scale-only and affine depth corrections.
        # -----------------------------------------------------------
        try:
            # Reject degenerate depth distributions before fitting.
            z_uni_range = Z_uni.max() - Z_uni.min()
            if z_uni_range < 5.0 or (z_uni_range / (Z_uni.max() + 1e-5)) < 0.1:
                debug_data["status"] = "Degenerate"
                return 1.0, debug_data

            X = Z_uni.reshape(-1, 1)
            y = Z_target # (N,) 1D array

            
            ransac_rigid = linear_model.RANSACRegressor(
                estimator=linear_model.LinearRegression(fit_intercept=False), 
                min_samples=15, max_trials=100, residual_threshold=1.0 
            )
            ransac_rigid.fit(X, y)
            
            
            s_rigid = ransac_rigid.estimator_.coef_[0] 
            inlier_ratio_rigid = np.sum(ransac_rigid.inlier_mask_) / len(y)

            
            ransac_affine = linear_model.RANSACRegressor(
                estimator=linear_model.LinearRegression(fit_intercept=True),
                min_samples=15, max_trials=100, residual_threshold=0.5
            )
            ransac_affine.fit(X, y)
            
            
            s_affine = ransac_affine.estimator_.coef_[0]
            bias_affine = ransac_affine.estimator_.intercept_ 
            inlier_ratio_affine = np.sum(ransac_affine.inlier_mask_) / len(y)

            
            final_s = 1.0
            used_model = "None"
            
            bias_is_bad = abs(bias_affine) > 5.0
            # if (inlier_ratio_affine > inlier_ratio_rigid + 0.1) and not bias_is_bad:
            if (inlier_ratio_affine > inlier_ratio_rigid) and not bias_is_bad:
                final_s = s_affine
                used_model = "Affine"
                debug_data["ransac_line"] = (s_affine, bias_affine)
            else:
                final_s = s_rigid
                used_model = "Rigid"
                debug_data["ransac_line"] = (s_rigid, 0.0)

            if final_s < 0.3 or final_s > 4.0:
                print(f"[ScaleWarn {frame_id_for_debug}] Ratio {final_s:.3f} Out of Bounds. Clamped.")
                final_s = np.clip(final_s, 0.3, 4)
                debug_data["status"] = f"Clamped {final_s:.2f}"
            else:
                print(f"[ScaleDebug {frame_id_for_debug}] CORRECTED! Model: {used_model}. Scale: {final_s:.4f}")
                debug_data["status"] = f"{used_model} {final_s:.2f}"

            return float(final_s), debug_data

        except Exception as e:
            
            import traceback
            traceback.print_exc()
            print(f"RANSAC Error: {e}")
            return 1.0, debug_data

    def check_pose_quality(self, R_curr, t_curr, inliers_count, frame_id):
        from utils.hybrid_motion import assess_motion
        prior = None
        if hasattr(self, 'prev_relative_R'):
            prior = (self.prev_relative_R, self.prev_relative_t)
        ids = getattr(self, 'accepted_frame_ids', [])
        ratio = (frame_id - ids[-1]) / (ids[-1] - ids[-2]) if len(ids) >= 2 else 1.0
        bad, reasons, info = assess_motion(
            R_curr, t_curr, inliers_count, self.track_cfg.get('hybrid', {}), prior, ratio)
        self.hybrid_decision = dict(info, reasons=reasons, triggered=bad)
        return bad, '; '.join(reasons)

    def _hybrid_refine(self, frame_rgb, depth, frame_id, R, t, scale, inliers, track_mode):
        import json
        import time
        from utils.hybrid_motion import valid_depth_matches, reprojection_metrics
        cfg = self.track_cfg.get('hybrid', {})
        metric_t = t * scale
        bad, reason = self.check_pose_quality(R, metric_t, inliers, frame_id)
        record = dict(self.hybrid_decision, frame_idx=int(frame_id), accepted=False,
                      reference_frame_idx=int(self.accepted_frame_ids[-1]),
                      base_R=R.tolist(), base_t=metric_t.reshape(3).tolist())
        if bad:
            tick = time.perf_counter()
            Log(f"[Hybrid] Frame {frame_id}: {reason}")
            kp_ref, kp_cur = self.heavy_matching(frame_rgb, self.prev_rgb, depth)
            kp_ref, kp_cur, _ = valid_depth_matches(kp_ref, kp_cur, self.prev_depth)
            record['matches'] = len(kp_ref)
            record['matching'] = self.heavy_matcher.last_stats
            if len(kp_ref) >= cfg.get('min_matches', 24):
                solver = cfg.get('solver', 'pnp')
                Rh, th, nh = compute_pose_3d2d_parallel(
                    kp_ref, kp_cur, self.prev_depth, self.cam,
                    max_ransac_iter=self.config['SLAM']['3d2d_thread'],
                    reprojection_error=cfg.get('pnp_reprojection_error_px', 1.0))
                candidate_solver = 'pnp'
                if solver == 'essential':
                    Re, te, ne = compute_pose_2d2d_parallel(
                        kp_ref, kp_cur, self.cam,
                        max_ransac_iter=self.config['SLAM']['2d2d_thread'],
                        essential_threshold=cfg.get('essential_threshold_px', 0.2),
                        homography_threshold=cfg.get('homography_threshold_px', 1.0),
                        gric_sigma=cfg.get('gric_sigma_px', 0.8))
                    transform = np.eye(4)
                    transform[:3, :3], transform[:3, 3:] = Re, te
                    se = find_scale_from_depth(self.cam, kp_ref, kp_cur, np.linalg.inv(transform), depth)
                    if se > 0 and np.isfinite(se) and np.linalg.norm(te) > 0 and ne >= cfg.get('min_candidate_inliers', 30):
                        Rh, th, nh, candidate_solver = Re, te * se, ne, 'essential'
                elif solver != 'pnp':
                    raise ValueError(f"Unknown hybrid solver: {solver}")
                base = reprojection_metrics(R, metric_t, kp_ref, kp_cur, self.prev_depth, self.K)
                candidate = reprojection_metrics(Rh, th, kp_ref, kp_cur, self.prev_depth, self.K)
                valid = (np.isfinite(Rh).all() and np.isfinite(th).all()
                         and nh >= cfg.get('min_candidate_inliers', 30)
                         and nh / len(kp_ref) >= cfg.get('min_inlier_ratio', 0.15))
                # A successful PnP can represent pure rotation (zero translation).
                # Failed PnP is rejected by its zero inlier count, not by |t| alone.
                if cfg.get('acceptance', 'support') == 'support':
                    valid = valid and candidate['support'] >= cfg.get('min_support', 0.25)
                    valid = valid and candidate['support'] >= base['support'] + cfg.get('support_gain', 0.05)
                record.update(candidate_solver=candidate_solver, candidate_inliers=int(nh),
                              candidate_R=Rh.tolist(), candidate_t=th.reshape(3).tolist(),
                              base_reprojection=base, candidate_reprojection=candidate,
                              eligible=bool(valid),
                              accepted=bool(valid and cfg.get("apply_replacement", True)))
                if record["accepted"]:
                    R, t, scale, inliers, track_mode = Rh, th, 1.0, nh, 2
                    self.diagnoser.log_extra(frame_id, 'MASt3R_Accepted', reason)
            record['heavy_seconds'] = time.perf_counter() - tick
        record.update(selected_R=R.tolist(), selected_t=(t * scale).reshape(3).tolist())
        with open(os.path.join(self.save_dir, 'hybrid_events.jsonl'), 'a') as f:
            f.write(json.dumps(record) + '\n')
        return R, t, scale, inliers, track_mode

    def motion_estimation(self, frame, frame_rgb, depth, frame_id, true_frame_id, flow_mag, last_img, last_depth, sky_mask=None):
        """
        Estimates the motion from current frame and computed keypoints

        Args:
            frame {ndarray}: frame to be processed
            frame_id {int}: integer corresponding to the frame id
            pose {list}: list with ground truth pose [x, y, z]
        """


        # print(f"DEBUG: Mode={self.feature_mode}, HeavyModel={self.heavy_model is not None}")
        
        self.feat_ref, self.feat_curr = self.feature_matching(frame, frame_rgb, depth = depth, filter=self.config["tracking"]["filter"], sky_mask=sky_mask, depth_threshold=65)

        # -------------------------------------------------------------
        
        # 0: E-Tracker (2D-2D), 1: PnP-Tracker (3D-2D)
        track_mode = 0 
        current_scale = 1.0
        # -------------------------------------------------------------

        # Estimate Essential geometry and compare it with Homography using GRIC.


        # Essential translation initially provides direction only.
        R, t, inliers_2d = compute_pose_2d2d_parallel(self.feat_ref, 
                                          self.feat_curr, 
                                          self.cam, 
                                          max_ransac_iter=self.config['SLAM']['2d2d_thread'])
        # Filter depth before scale recovery and PnP.


        s = 1.0

        depth_raw_for_viz = depth.copy()        
        depth = preprocess_depth(depth, crop=[[0.3, 1], [0, 1]], depth_range=[0, 100], sky_mask=sky_mask)
        
        # self.diagnoser.visualize_scale_debug(true_frame_id, depth_raw_for_viz, depth, frame_rgb, self.prev_frame, debug_info)

        cands = None            
        E_pose = np.eye(4)      
        
        
        if frame_id == 1:
            # Initialize the first relative motion through the Essential path.
            self.R = R
            self.t = t
            
            E_pose[:3, :3] = R
            E_pose[: 3, 3:] = t


        else:
            
            E_pose = np.eye(4)
            E_pose[:3, :3] = R
            E_pose[: 3, 3:] = t

            # Retrieve loop candidates asynchronously, then verify geometry.
            if frame_id > 10 and self.loop_enable:
                
                
                cands = self.retrieval.detect_loop(thresh=self.config['SLAM']['loop_detect_thresh'], num_repeat=3)

                if cands is not None:
                    Log(f' -----> {cands} LOOP DETECTED!')
                    (i, j) = cands      
                    
                    # Re-match the loop candidate with lightweight features.
                    feat_ref_loop, feat_curr_loop = self.feature_matching(self.frame_history[i], 
                                                                            frame_rgb,
                                                                            depth = self.depth_history[i], 
                                                                            lc_frame_id = j, 
                                                                            depth_threshold=55)

                    
                    # Back-project matched points for Sim(3) loop verification.
                    pts_world1, pts_world2 = get_matched_camera_points_vectorized(
                        depth_map1=self.depth_history[i], feat_points1=feat_curr_loop,
                        depth_map2=self.depth_history[j], feat_points2=feat_ref_loop,
                        K=self.K
                    )
                    
                    if self.config['SLAM']['viz']:
                        filename = os.path.join(self.save_dir, 'loop_pair_viz', f'loop_{i}_{j}.png')
                        draw_concat_keypoints(self.frame_history[i], feat_curr_loop,
                                              self.frame_history[j], feat_ref_loop,
                                              filename)
                    
                    
                    self.retrieval.confirm_loop(i, j)
                    self.retrieval.found.clear()
                
                self.retrieval.save_up_to(frame_id)

            # --- Motion Estimation Logic ---
            if cands is None or not self.loop_enable:

                # Recover metric translation scale from predicted depth.
                
                
                scale = find_scale_from_depth(
                    self.cam,
                    self.feat_ref,
                    self.feat_curr,
                    np.linalg.inv(E_pose), # reference -> current, as required by triangulation
                    depth
                )
                
                
                current_scale = scale

                
                inliers_pnp = 0
                if np.linalg.norm(t) == 0 or scale == -1.0:
                    
                    
                    track_mode = 1
                    if scale == -1.0:
                        Log(f"[Tracker] Frame {frame_id}: Scale recovery failed, fallback to PnP.")
                    else:
                        Log(f"[Tracker] Frame {frame_id}: Pure rotation detected (t=0), fallback to PnP.")


                    R, t, inliers_pnp = compute_pose_3d2d_parallel(
                                self.feat_ref,
                                self.feat_curr,
                                self.prev_depth, 
                                self.cam,
                                max_ransac_iter=self.config['SLAM']['3d2d_thread']
                            )  # pose: from cur->ref
                    
                
                scale = 1 if scale == -1.0 else scale
                
                # =========================================================================
                
                # =========================================================================
                if self.feature_mode == 'hybrid':
                    base_inliers = inliers_pnp if track_mode == 1 else inliers_2d
                    R, t, scale, selected_inliers, track_mode = self._hybrid_refine(
                        frame_rgb, depth, true_frame_id, R, t, scale, base_inliers, track_mode)
                    if track_mode == 2:
                        inliers_pnp = selected_inliers

                
                self.prev_relative_R = R
                self.prev_relative_t = t * scale 

                self.t = self.t + scale * self.R.dot(t)
                
                self.R = self.R.dot(R)

            else:
                
                
                R, t, _ = compute_pose_2d2d_parallel(feat_ref_loop, 
                                                  feat_curr_loop, 
                                                  self.cam, 
                                                  max_ransac_iter=self.config['SLAM']['2d2d_thread'])


                loop_r = R
                loop_t = t.squeeze()
                loop_s = 1.0        
                far_rel_pose = make_pypose_Sim3(loop_r, loop_t, loop_s)[None]   
                
                
                poses_quat = torch.tensor(convert_pose_numpy_to_opencv_vectorized(self.pose_history, inv = True)).unsqueeze(0) 
                
                Gi = pp.SE3(poses_quat[:,self.loop_ii])
                Gj = pp.SE3(poses_quat[:,self.loop_jj])
                Gij = Gj * Gi.Inv() 
                prev_sim3 = SE3_to_Sim3(Gij).data[0].cpu() 
                
                loop_poses = pp.Sim3(torch.cat((prev_sim3, far_rel_pose))) 
                
                loop_ii = torch.cat((self.loop_ii, torch.tensor([i]))) 
                loop_jj = torch.cat((self.loop_jj, torch.tensor([j]))) 
                
                pred_poses = pp.SE3(poses_quat.squeeze()[:frame_id]).Inv().cpu() 
                
                self.loop_ii = loop_ii
                self.loop_jj = loop_jj
                
                
                final_est = pose_refinement(pred_poses.data.to(torch.float32), loop_poses.data.to(torch.float32), loop_ii, loop_jj)
                
                # Align the corrected trajectory gauge to the current reference.
                
                
                safe_i = final_est.shape[0] - 1
                aa = SE3_to_Sim3(pred_poses.cpu()).to(torch.float32)
                final_est = (aa[[safe_i]] * final_est[[safe_i]].Inv()) * final_est
                output = final_est
                
                output_matrix = poses_to_c2w_tensor(output[:, :7]).numpy()
                
                E_pose_loop = output_matrix[-1, :, :]
                t_loop = E_pose_loop[:3, 3:]
                R_loop = E_pose_loop[:3, :3]
                self.t = t_loop
                self.R = R_loop
                
                
                # Retain old poses so mapping can transform existing anchors.
                self.pose_history_old = self.pose_history.copy()
                self.pose_history[:output_matrix.shape[0], :, :] = output_matrix
                
                self.loop_kf_idx = output_matrix.shape[0]
                self.viz_pose(frame_id)

        # -------------------------------------------------------------
        
        # mode: 0=E, 1=PnP


        # -------------------------------------------------------------
        self.diagnoser.log(
            frame_id=true_frame_id,
            mode="MASt3R" if track_mode == 2 else ("PnP" if track_mode == 1 else "E"),
            scale=current_scale,
            inliers=inliers_2d if track_mode == 0 else inliers_pnp,
            flow_mag=flow_mag, 
            has_loop=(cands is not None)
        )

        
        if (true_frame_id >= self.config['SLAM']['frame_num'] - 5) or (true_frame_id % 100 == 0):
            self.diagnoser.save_and_plot()
        
        # -------------------------------------------------------------
        
        
        E_pose[:3, :3] = self.R
        E_pose[: 3, 3:] = self.t
        self.prev_depth = depth     


        if self.feat_ref.shape[0] < self.min_num_features:
            self.feat_curr = self.detect_features(frame)
            self.feat_curr = np.array([x.pt for x in self.feat_curr], dtype=np.float32)

        
        self.feat_ref = self.feat_curr
        self.pose_ref = E_pose
        
        self.pose_history[frame_id] = E_pose
        self.depth_history.append(depth)
        self.kp_history.append(self.feat_curr)
        self.frame_history.append(frame)

        if cands is None or not self.loop_enable:
            return s, False, None
        else:
            return s, True, cands
    # Advance tracking and return the current W2C pose for mapping.
    def update(self, frame, depth, frame_id, true_frame_id, last_img, last_depth, flow_mag=0.0, sky_mask=None):
        """
        Computes the camera motion between the current image and the previous one.
        frame shape: np.array([3, h, w]) and would be convert to [h, w]
        """


        frame_rgb = frame
        frame_normalized = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
        frame_uint8 = frame_normalized.astype(np.uint8)
        frame = cv2.cvtColor(np.transpose(frame_uint8, (1, 2, 0)), cv2.COLOR_RGB2GRAY)

        loop_flag = False
        
        if frame_id == 0:
            # Initialize the reference on the first frame.
            self.motion_estimation_init(frame)
            s=1
            cands = None
        else:

            s, loop_flag, cands = self.motion_estimation(frame, frame_rgb, depth, frame_id, true_frame_id, flow_mag, last_img, last_depth, sky_mask)


        self.prev_frame = frame
        self.prev_rgb = frame_rgb
        if not hasattr(self, 'accepted_frame_ids'):
            self.accepted_frame_ids = []
        self.accepted_frame_ids.append(int(true_frame_id))


        pose = np.eye(4)
        pose[:3, :3] = self.R
        pose[:3, 3:] = self.t
        self.pose = pose
        # Convert C2W to W2C for mapping and rendering.
        self.w2c = np.linalg.inv(pose)

        return loop_flag, s, cands
    
    def get_pose(self):
        return torch.from_numpy(self.w2c[:3, :3]), torch.from_numpy(self.w2c[:3, 3:]).squeeze()
    
    def update_tracking_pose(self, c2w):
        self.pose = c2w
        self.w2c = np.linalg.inv(self.pose)
        

def preprocess_depth(depth, crop, depth_range, sky_mask=None):
    """
    Args:
        depth: [H, W] NumPy depth map.
        crop: normalized [[y_min, y_max], [x_min, x_max]] bounds.
        depth_range: [min_depth, max_depth]
        sky_mask: optional [H, W] mask; 1 marks valid objects, 0 marks sky.
    """
    min_depth, max_depth = depth_range
    h, w = depth.shape


    valid_base_mask = (depth > min_depth)

    if sky_mask is not None:
        # With a sky mask, use validity and positive depth instead of crop/range filtering.


        binary_sky_valid = (sky_mask > 0.5) 
        
        valid_mask = valid_base_mask & binary_sky_valid
        
    else:
        # Without a sky mask, apply the crop and depth-range limits.
        
        y0, y1 = int(h * crop[0][0]), int(h * crop[0][1])
        x0, x1 = int(w * crop[1][0]), int(w * crop[1][1])
        crop_mask = np.zeros((h, w), dtype=bool)
        crop_mask[y0:y1, x0:x1] = True

        
        range_max_mask = (depth < max_depth)

        
        valid_mask = valid_base_mask & crop_mask & range_max_mask


    depth = depth * valid_mask.astype(depth.dtype)
    
    return depth


def compute_pose_3d2d(kp1, kp2, depth_1, cam_intrinsics):
    """
    https://github.com/Huangying-Zhan/DF-VO
    """
    max_depth = 50
    min_depth = 0

    outputs = {}
    height, width = depth_1.shape

    # Filter keypoints outside image region
    x_idx = (kp1[:, 0] >= 0) * (kp1[:, 0] < width)
    kp1 = kp1[x_idx]
    kp2 = kp2[x_idx]
    x_idx = (kp2[:, 0] >= 0) * (kp2[:, 0] < width)
    kp1 = kp1[x_idx]
    kp2 = kp2[x_idx]
    y_idx = (kp1[:, 1] >= 0) * (kp1[:, 1] < height)
    kp1 = kp1[y_idx]
    kp2 = kp2[y_idx]
    y_idx = (kp2[:, 1] >= 0) * (kp2[:, 1] < height)
    kp1 = kp1[y_idx]
    kp2 = kp2[y_idx]

    # Filter keypoints outside depth range
    kp1_int = kp1.astype(int)
    kp_depths = depth_1[kp1_int[:, 1], kp1_int[:, 0]]
    non_zero_mask = (kp_depths != 0)
    depth_range_mask = (kp_depths < max_depth) * (kp_depths > min_depth)
    valid_kp_mask = non_zero_mask * depth_range_mask

    kp1 = kp1[valid_kp_mask]
    kp2 = kp2[valid_kp_mask]

    # Get 3D coordinates for kp1
    XYZ_kp1 = unprojection_kp(kp1, kp_depths[valid_kp_mask], cam_intrinsics)

    # initialize ransac setup
    best_rt = []
    best_inlier = 0
    max_ransac_iter = 3

    for _ in range(max_ransac_iter):
        # shuffle kp (only useful when random seed is fixed)
        new_list = np.arange(0, kp2.shape[0], 1)
        np.random.shuffle(new_list)
        new_XYZ = XYZ_kp1.copy()[new_list]
        new_kp2 = kp2.copy()[new_list]

        if new_kp2.shape[0] > 4:
            # PnP solver
            flag, r, t, inlier = cv2.solvePnPRansac(
                objectPoints=new_XYZ,
                imagePoints=new_kp2,
                cameraMatrix=cam_intrinsics.mat,
                distCoeffs=None,
                iterationsCount=100,  # number of iteration
                reprojectionError=1,  # inlier threshold value
            )

            # save best pose estimation
            if flag and inlier.shape[0] > best_inlier:
                best_rt = [r, t]
                best_inlier = inlier.shape[0]

    # format pose
    R = np.eye(3)
    t = np.zeros((3, 1))
    if len(best_rt) != 0:
        r, t = best_rt
        R = cv2.Rodrigues(r)[0]
    E_pose = np.eye(4)
    E_pose[:3, :3] = R
    E_pose[: 3, 3:] = t
    E_pose = np.linalg.inv(E_pose)
    R = E_pose[:3, :3]
    t = E_pose[: 3, 3:]
    return R, t


def compute_pose_3d2d_parallel(kp1, kp2, depth_1, cam_intrinsics, max_ransac_iter=20, reprojection_error=1.0):
    '''
        Faster
    '''
    max_depth = 50
    min_depth = 0

    height, width = depth_1.shape

    x_idx = (kp1[:, 0] >= 0) & (kp1[:, 0] < width) & (kp2[:, 0] >= 0) & (kp2[:, 0] < width)
    kp1 = kp1[x_idx]
    kp2 = kp2[x_idx]
    y_idx = (kp1[:, 1] >= 0) & (kp1[:, 1] < height) & (kp2[:, 1] >= 0) & (kp2[:, 1] < height)
    kp1 = kp1[y_idx]
    kp2 = kp2[y_idx]

    kp1_int = kp1.astype(int)
    kp_depths = depth_1[kp1_int[:, 1], kp1_int[:, 0]]
    valid_kp_mask = (kp_depths != 0) & (kp_depths < max_depth) & (kp_depths > min_depth)
    kp1 = kp1[valid_kp_mask]
    kp2 = kp2[valid_kp_mask]
    kp_depths = kp_depths[valid_kp_mask]

    if len(kp1) < 5:
        return np.eye(3), np.zeros((3, 1)), 0  # Not enough points

    XYZ_kp1 = unprojection_kp(kp1, kp_depths, cam_intrinsics)

    def run_pnp_once(seed=None):
        new_list = np.random.RandomState(seed).permutation(kp2.shape[0])
        cv2.setRNGSeed(int(seed or 0))
        new_XYZ = XYZ_kp1[new_list]
        new_kp2 = kp2[new_list]

        if new_kp2.shape[0] > 4:
            flag, r, t, inliers = cv2.solvePnPRansac(
                objectPoints=new_XYZ,
                imagePoints=new_kp2,
                cameraMatrix=cam_intrinsics.mat,
                distCoeffs=None,
                iterationsCount=100,
                reprojectionError=reprojection_error,
            )
            if flag and inliers is not None:
                return {
                    "inliers": len(inliers),
                    "r": r,
                    "t": t
                }
        return {"inliers": 0}

    with ThreadPoolExecutor(max_workers=max_ransac_iter) as executor:
        futures = [executor.submit(run_pnp_once, seed=i) for i in range(max_ransac_iter)]
        results = [f.result() for f in futures]

    best = max(results, key=lambda x: x["inliers"])
    if best["inliers"] == 0:
        return np.eye(3), np.zeros((3, 1)), 0  # fallback

    R = cv2.Rodrigues(best["r"])[0]
    t = best["t"]

    E_pose = np.eye(4)
    E_pose[:3, :3] = R
    E_pose[:3, 3:] = t
    E_pose = np.linalg.inv(E_pose)
    R = E_pose[:3, :3]
    t = E_pose[:3, 3:]

    return R, t, best["inliers"]


def unprojection_kp(kp, kp_depth, cam_intrinsics):
    """
    https://github.com/Huangying-Zhan/DF-VO
    """
    N = kp.shape[0]
    # initialize regular grid
    XYZ = np.ones((N, 3, 1))
    XYZ[:, :2, 0] = kp

    inv_K = np.ones((1, 3, 3))
    inv_K[0] = np.linalg.inv(cam_intrinsics.mat)  # cam_intrinsics.inv_mat
    inv_K = np.repeat(inv_K, N, axis=0)

    XYZ = np.matmul(inv_K, XYZ)[:, :, 0]
    XYZ[:, 0] = XYZ[:, 0] * kp_depth
    XYZ[:, 1] = XYZ[:, 1] * kp_depth
    XYZ[:, 2] = XYZ[:, 2] * kp_depth
    return XYZ


def compute_fundamental_residual(F, kp1, kp2):
    # get homogeneous keypoints (3xN array)
    m0 = np.ones((3, kp1.shape[0]))
    m0[:2] = np.transpose(kp1, (1, 0))
    m1 = np.ones((3, kp2.shape[0]))
    m1[:2] = np.transpose(kp2, (1, 0))

    Fm0 = F @ m0  # 3xN
    Ftm1 = F.T @ m1  # 3xN

    # Only the diagonal is needed. Avoid an N x N temporary for dense MASt3R matches.
    m1Fm0 = np.sum(Fm0 * m1, axis=0)
    res = m1Fm0 ** 2 / (np.sum(Fm0[:2] ** 2, axis=0) + np.sum(Ftm1[:2] ** 2, axis=0))
    return res

# Compute the first-order Sampson approximation for Homography residuals.


def compute_homography_residual(H_in, kp1, kp2):
    n = kp1.shape[0]
    H = H_in.flatten()

    # get homogeneous keypoints (3xN array)
    m0 = np.ones((3, kp1.shape[0]))
    m0[:2] = np.transpose(kp1, (1, 0))
    m1 = np.ones((3, kp2.shape[0]))
    m1[:2] = np.transpose(kp2, (1, 0))

    G0 = np.zeros((3, n))
    G1 = np.zeros((3, n))

    G0[0] = H[0] - m1[0] * H[6]
    G0[1] = H[1] - m1[0] * H[7]
    G0[2] = -m0[0] * H[6] - m0[1] * H[7] - H[8]

    G1[0] = H[3] - m1[1] * H[6]
    G1[1] = H[4] - m1[1] * H[7]
    G1[2] = -m0[0] * H[6] - m0[1] * H[7] - H[8]

    magG0 = np.sqrt(G0[0] * G0[0] + G0[1] * G0[1] + G0[2] * G0[2])
    magG1 = np.sqrt(G1[0] * G1[0] + G1[1] * G1[1] + G1[2] * G1[2])
    magG0G1 = G0[0] * G1[0] + G0[1] * G1[1]

    alpha = np.arccos(magG0G1 / (magG0 * magG1))

    alg = np.zeros((2, n))
    alg[0] = m0[0] * H[0] + m0[1] * H[1] + H[2] - \
             m1[0] * (m0[0] * H[6] + m0[1] * H[7] + H[8])

    alg[1] = m0[0] * H[3] + m0[1] * H[4] + H[5] - \
             m1[1] * (m0[0] * H[6] + m0[1] * H[7] + H[8])

    D1 = alg[0] / magG0
    D2 = alg[1] / magG1

    res = (D1 * D1 + D2 * D2 - 2.0 * D1 * D2 * np.cos(alpha)) / np.sin(alpha)

    return res

# GRIC balances geometric residuals and model complexity; lower is better.

def calc_GRIC(res, sigma, n, model):
    """
    Calculate GRIC
    """
    R = 4
    sigmasq1 = 1. / sigma ** 2

    K = {
        "FMat": 7,
        "EMat": 5,
        "HMat": 8,
    }[model]
    D = {
        "FMat": 3,
        "EMat": 3,
        "HMat": 2,
    }[model]

    lam3RD = 2.0 * (R - D)

    sum_ = 0
    for i in range(n):
        tmp = res[i] * sigmasq1
        if tmp <= lam3RD:
            sum_ += tmp
        else:
            sum_ += lam3RD

    sum_ += n * D * np.log(R) + K * np.log(R * n)
    
    
    return sum_

def compute_pose_2d2d_parallel(kp_ref, kp_cur, cam_intrinsics, max_ransac_iter=20, essential_threshold=0.2, homography_threshold=1.0, gric_sigma=0.8):
    '''
        Faster
    '''
    
    
    principal_points = (cam_intrinsics.cx, cam_intrinsics.cy)
    
    if kp_cur.shape[0] <= 10:
        return np.eye(3), np.zeros((3, 1)), 0
    
    
    H, H_inliers = cv2.findHomography(
        kp_cur, kp_ref,
        method=cv2.RANSAC,
        confidence=0.99,
        ransacReprojThreshold=homography_threshold,
    )
    if H is None or not np.isfinite(H).all():
        # Degenerate correspondences must reach the existing PnP/base fallback.
        return np.eye(3), np.zeros((3, 1)), 0
    
    H_res = compute_homography_residual(H, kp_cur, kp_ref)
    
    
    H_gric = calc_GRIC(
        res=H_res,
        sigma=gric_sigma,
        n=kp_cur.shape[0],
        model="HMat"
    )
    
    
    def ransac_worker(seed):
        
        indices = np.random.RandomState(seed).permutation(kp_cur.shape[0])
        cv2.setRNGSeed(int(seed))

        new_kp_cur = kp_cur[indices]
        new_kp_ref = kp_ref[indices]
        
        
        E, inliers = cv2.findEssentialMat(
            new_kp_cur,
            new_kp_ref,
            focal=cam_intrinsics.fx,
            pp=principal_points,
            method=cv2.RANSAC,
            prob=0.99,
            threshold=essential_threshold,
        )

        if E is None:
            return None
        # Convert E to pixel-coordinate F before computing residuals.
        # Normalize pixel coordinates with the camera intrinsics.
        
        K = cam_intrinsics.mat
        F = np.linalg.inv(K.T) @ E @ np.linalg.inv(K)
        
        E_res = compute_fundamental_residual(F, new_kp_cur, new_kp_ref)
        
        
        E_gric = calc_GRIC(res=E_res, sigma=gric_sigma, n=kp_cur.shape[0], model='EMat')
        
        # Reject Essential candidates when Homography has a lower GRIC score.
        
        
        if H_gric <= E_gric:
            return None 

        inlier_count = int(inliers.sum())
        revert_indices = np.argsort(indices)
        final_inliers = inliers[revert_indices]
        
        return {
            'E': E,
            'inliers': final_inliers,
            'count': inlier_count
        }
    
    # Run independent RANSAC attempts and retain the strongest supported candidate.
    with ThreadPoolExecutor(max_workers=max_ransac_iter) as executor:
        results = list(executor.map(ransac_worker, range(max_ransac_iter)))
    
    best_result = max([r for r in results if r is not None], key=lambda x: x['count'], default=None)
    
    if best_result is None:
        return np.eye(3), np.zeros((3, 1)), 0


    # Cheirality requires triangulated points in front of both cameras.
    cheirality_cnt, R, t, _ = cv2.recoverPose(
        best_result['E'],
        kp_cur,
        kp_ref,
        focal=cam_intrinsics.fx,
        pp=principal_points
    )

    # Reject poses with insufficient cheirality support.
    if cheirality_cnt > kp_cur.shape[0] * 0.1:
        return R, t, best_result['count']
    else:
        return np.eye(3), np.zeros((3, 1)), 0


def compute_pose_2d2d(kp_ref, kp_cur, cam_intrinsics):
    """
    https://github.com/Huangying-Zhan/DF-VO
    """
    principal_points = (cam_intrinsics.cx, cam_intrinsics.cy)

    # initialize ransac setup
    R = np.eye(3)
    t = np.zeros((3, 1))
    best_Rt = [R, t]
    best_inlier_cnt = 0
    max_ransac_iter = 5
    best_inliers = np.ones((kp_ref.shape[0], 1)) == 1

    # method GRIC of validating E-tracker
    if kp_cur.shape[0] > 10:


        H, H_inliers = cv2.findHomography(
            kp_cur,
            kp_ref,
            method=cv2.RANSAC,
            confidence=0.99,
            ransacReprojThreshold=1,
        )

        H_res = compute_homography_residual(H, kp_cur, kp_ref)
        H_gric = calc_GRIC(
            res=H_res,
            sigma=0.8,
            n=kp_cur.shape[0],
            model="HMat"
        )

        valid_case = True
    else:
        valid_case = False

    if valid_case:
        num_valid_case = 0
        for i in range(max_ransac_iter):  # repeat ransac for several times for stable result
            # shuffle kp_cur and kp_ref (only useful when random seed is fixed)
            new_list = np.arange(0, kp_cur.shape[0], 1)
            np.random.shuffle(new_list)
            new_kp_cur = kp_cur.copy()[new_list]
            new_kp_ref = kp_ref.copy()[new_list]

            E, inliers = cv2.findEssentialMat(
                new_kp_cur,
                new_kp_ref,
                focal=cam_intrinsics.fx,
                pp=principal_points,
                method=cv2.RANSAC,

                prob=0.99,
                threshold=0.2,
            )


            # get F from E
            K = cam_intrinsics.mat
            F = np.linalg.inv(K.T) @ E @ np.linalg.inv(K)
            E_res = compute_fundamental_residual(F, new_kp_cur, new_kp_ref)

            E_gric = calc_GRIC(
                res=E_res,
                sigma=0.8,
                n=kp_cur.shape[0],
                model='EMat'
            )
            valid_case = H_gric > E_gric

            # inlier check
            inlier_check = inliers.sum() > best_inlier_cnt

            # save best_E
            if inlier_check:
                best_E = E
                best_inlier_cnt = inliers.sum()

                revert_new_list = np.zeros_like(new_list)
                for cnt, i in enumerate(new_list):
                    revert_new_list[i] = cnt
                best_inliers = inliers[list(revert_new_list)]
            num_valid_case += (valid_case * 1)

        major_valid = num_valid_case > (max_ransac_iter / 2)
        
        if major_valid:
            cheirality_cnt, R, t, _ = cv2.recoverPose(best_E, kp_cur, kp_ref,
                                                      focal=cam_intrinsics.fx,
                                                      pp=principal_points,
                                                      )

            # cheirality_check
            if cheirality_cnt > kp_cur.shape[0] * 0.1:
                best_Rt = [R, t]

    R, t = best_Rt
    return R, t
