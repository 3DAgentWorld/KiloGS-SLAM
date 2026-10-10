import time
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import cv2

from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2
from utils.camera_utils import Camera
from utils.logging_utils import Log
from utils.slam_utils import update_viewpoints_from_poses

from utils.visual_odometry import ClassicTracking

from unidepth.models import UniDepthV2 


import huggingface_hub

class FrontEnd(mp.Process):
    def __init__(self, config, dataset, save_dir = None):
        super().__init__()
        self.dataset = dataset
        self.save_dir = save_dir

        self.config = config
        self.background = None
        self.pipeline_params = None
        self.frontend_queue = None
        self.backend_queue = None
        self.q_main2vis = None
        self.q_vis2main = None

        self.initialized = False
        self.kf_indices = []
        self.monocular = config["Training"]["monocular"]
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.current_window = []

        self.reset = True
        self.requested_init = False
        self.requested_keyframe = 0
        self.use_every_n_frames = 1
        self.last_img = None
        self.last_depth = None

        self.gaussians = None
        self.cameras = dict()
        self.device = "cuda:0"
        self.pause = False

        cam_params = {}
        cam_params['fx'] = self.dataset.fx,
        cam_params['fy'] = self.dataset.fy,
        cam_params['cx'] = self.dataset.cx,
        cam_params['cy'] = self.dataset.cy,
        cam_params['width'] = self.dataset.width,
        cam_params['height'] = self.dataset.height,

        self.width = float(cam_params['width'][0])
        self.height = float(cam_params['height'][0])
        self.fx = float(cam_params['fx'][0])
        self.fy = float(cam_params['fy'][0])
        self.cx = float(cam_params['cx'][0])
        self.cy = float(cam_params['cy'][0])

        self.intrinsics = torch.tensor([
                [self.fx,   0,          self.cx ],
                [0,         self.fy,    self.cy ],
                [0,         0,          1       ]
            ], device='cuda')
        
        cam_params_ori = {}
        cam_params_ori['fx'] = self.dataset.fx_ori,
        cam_params_ori['fy'] = self.dataset.fy_ori,
        cam_params_ori['cx'] = self.dataset.cx_ori,
        cam_params_ori['cy'] = self.dataset.cy_ori,
        cam_params_ori['width'] = self.dataset.width_ori,
        cam_params_ori['height'] = self.dataset.height_ori,

        self.intrinsics_ori = torch.tensor([
                [self.dataset.fx_ori,   0,                      self.dataset.cx_ori ],
                [0,                     self.dataset.fy_ori,    self.dataset.cy_ori ],
                [0,                     0,                      1                   ]
            ], device='cuda')
        
        loop_enable = config['SLAM']['loop_closure']
        self.viz = config['SLAM']['viz']

        self.viz_frame_id_prev = 0
        self.viz_frame_id_curr = 0
        self.viz_frame_interval = 50

        self.classic_tracking = ClassicTracking(cam_params_ori, loop_enable, self.viz, config, save_dir = self.save_dir)

        self.motion_thresh = config['SLAM']['motion_thresh']

        self.keyframe_idx_list = []

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        depth_config = config['DepthModel']
        selected_model = depth_config.get('selected_model', 'UniDepthV2') 

        ## depth model init
        if selected_model == 'DepthAnything3':
            print(f"[FrontEnd] Selected Depth Model: Depth Anything 3")
            
            from utils.depth_anything_wrapper import DepthAnythingWrapper
            self.depth_model = DepthAnythingWrapper(depth_config['DepthAnything3'])

        else: 
            print(f"[FrontEnd] Selected Depth Model: UniDepth V2")
            ud_cfg = depth_config['UniDepthV2']
            if ud_cfg['from_huggingface']:
                self.depth_model = UniDepthV2.from_pretrained(
                    ud_cfg['huggingface']['model_name'],
                    revision=ud_cfg['huggingface']['commit_hash']
                )
            else:
                self.depth_model = UniDepthV2.from_pretrained(ud_cfg['local_snapshot_path'])
            
            self.depth_model = self.depth_model.to(device)

    def set_hyperparams(self):
        self.save_dir = self.config["Results"]["save_dir"]
        self.save_results = self.config["Results"]["save_results"]
        self.save_trj = self.config["Results"]["save_trj"]
        self.save_trj_kf_intv = self.config["Results"]["save_trj_kf_intv"]

        self.tracking_itr_num = self.config["Training"]["tracking_itr_num"]
        self.kf_interval = self.config["Training"]["kf_interval"]
        self.window_size = self.config["Training"]["window_size"]
        self.single_thread = self.config["Training"]["single_thread"]

    def add_new_keyframe(self, cur_frame_idx, init=False):
        rgb_boundary_threshold = self.config["Training"]["rgb_boundary_threshold"]
        self.kf_indices.append(cur_frame_idx)
        viewpoint = self.cameras[cur_frame_idx]
        gt_img = viewpoint.original_image.cuda()
        valid_rgb = (gt_img.sum(dim=0) > rgb_boundary_threshold)[None]

        initial_depth = viewpoint.depth
        initial_depth[~valid_rgb] = 0  # Ignore the invalid rgb pixels
        return initial_depth[0].to(torch.float32)

    def initialize(self, cur_frame_idx, viewpoint):
        self.initialized = not self.monocular
        self.kf_indices = []
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.current_window = []
        # remove everything from the queues
        while not self.backend_queue.empty():
            self.backend_queue.get()

        # Initialise the frame at the ground truth pose
        T = torch.eye(4, device=viewpoint.device)
        R_init = T[:3, :3]
        T_init = T[:3, 3]
        # viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)
        viewpoint.update_RT(R_init, T_init)

        self.kf_indices = []
        #self.kf_indices.append(cur_frame_idx)
        depth_map = self.add_new_keyframe(cur_frame_idx, init=True)
        self.request_init(cur_frame_idx, viewpoint, depth_map)
        self.reset = False

    def tracking(self, cur_frame_idx, viewpoint):

        R, T = self.classic_tracking.get_pose()

        viewpoint.update_RT(R, T)


    def is_keyframe(
        self,
        cur_frame_idx,
        last_keyframe_idx
    ):
        kf_translation = self.config["Training"]["kf_translation"]
        kf_min_translation = self.config["Training"]["kf_min_translation"]
        kf_overlap = self.config["Training"]["kf_overlap"]

        curr_frame = self.cameras[cur_frame_idx]
        last_kf = self.cameras[last_keyframe_idx]
        pose_CW = getWorld2View2(curr_frame.R, curr_frame.T)
        last_kf_CW = getWorld2View2(last_kf.R, last_kf.T)
        last_kf_WC = torch.linalg.inv(last_kf_CW)
        dist = torch.norm((pose_CW @ last_kf_WC)[0:3, 3])
        dist_check = dist > kf_translation * self.median_depth
        dist_check2 = dist > kf_min_translation * self.median_depth

        return dist_check2 or dist_check

    def add_to_window(
        self, cur_frame_idx, window
    ):
        window = [cur_frame_idx] + window
        # remove frames which has little overlap with the current frame

        removed_frame = None

        if len(window) > self.config["Training"]["window_size"]:


            removed_frame = window[-1]
            window.remove(removed_frame)

        return window, removed_frame

    def request_keyframe(self, cur_frame_idx, viewpoint, current_window, depthmap):
        Log(f'request_keyframe: {cur_frame_idx}')
        msg = ["keyframe", cur_frame_idx, viewpoint, current_window, depthmap]
        self.backend_queue.put(msg)
        self.requested_keyframe += 1

    def reqeust_mapping(self, cur_frame_idx, viewpoint):
        msg = ["map", cur_frame_idx, viewpoint]
        self.backend_queue.put(msg)

    def request_init(self, cur_frame_idx, viewpoint, depth_map):
        msg = ["init", cur_frame_idx, viewpoint, depth_map]
        self.backend_queue.put(msg)
        self.requested_init = True

    def request_loop_update(self, cur_frame_idx, poses_old, poses_update, kf_indices, loop_record):
        msg = ["loop_update", cur_frame_idx, poses_old, poses_update, kf_indices, loop_record]
        self.backend_queue.put(msg)

    def sync_backend(self, data):
        if data[1] is not None:
            self.gaussians = data[1]

        keyframes = data[3]

        self.last_img = data[4]
        self.last_depth = data[5]

        for kf_id, kf_R, kf_T in keyframes:
            self.cameras[kf_id].update_RT(kf_R.clone(), kf_T.clone())

    def cleanup(self, cur_frame_idx):
        self.cameras[cur_frame_idx].clean()
        if cur_frame_idx % 10 == 0:
            torch.cuda.empty_cache()

    def run(self):
        cur_frame_idx = 0
        projection_matrix = getProjectionMatrix2(
            znear=0.01,
            zfar=100.0,
            fx=self.dataset.fx,
            fy=self.dataset.fy,
            cx=self.dataset.cx,
            cy=self.dataset.cy,
            W=self.dataset.width,
            H=self.dataset.height,
        ).transpose(0, 1)
        projection_matrix = projection_matrix.to(device=self.device)
        tic = torch.cuda.Event(enable_timing=True)
        toc = torch.cuda.Event(enable_timing=True)

        while True:
            if self.q_vis2main.empty():
                if self.pause:
                    continue
            else:
                data_vis2main = self.q_vis2main.get()
                self.pause = data_vis2main.flag_pause
                if self.pause:
                    self.backend_queue.put(["pause"])
                    continue
                else:
                    self.backend_queue.put(["unpause"])

            if self.frontend_queue.empty():
                tic.record()
                if cur_frame_idx >= len(self.dataset): 
                    break

                if self.requested_init:
                    time.sleep(0.01)
                    continue

                if self.single_thread and self.requested_keyframe > 0:
                #if self.requested_keyframe > 0:
                    time.sleep(0.01)
                    continue

                if not self.initialized and self.requested_keyframe > 0:
                    time.sleep(0.01)
                    continue

                viewpoint = Camera.init_from_dataset(
                    self.dataset, cur_frame_idx, projection_matrix
                )
                # Skip frames with insufficient image motion.
                flow = 0.0
                if cur_frame_idx > 2:
                    flow = self.classic_tracking.motion_flow_keyframe(viewpoint.image_ori, prev_kf=False)
                    if flow < self.motion_thresh:
                        cur_frame_idx += 1

                        Log(f'Skip frame {cur_frame_idx}')
                        continue


                # # unidepth take [C(rgb), H, W] as input 0~255 in uint8
                # image_ori_for_depthmodel = torch.tensor(viewpoint.image_ori).cuda().to(torch.uint8)
                # predictions = self.depth_model.infer(image_ori_for_depthmodel, self.intrinsics_ori) # return [1, 1, H, W]
                # depth_pred = predictions["depth"].detach() # depth range: 0-255 in float
                # depth_ori = depth_pred.squeeze().cpu().numpy() # shape: [H, W]
                # depth_down = cv2.resize(depth_ori, (self.dataset.width, self.dataset.height))
                # viewpoint.depth_ori = depth_ori
                # viewpoint.depth =depth_down
                
                ## ==============depth inference==================
                # unidepth take [C(rgb), H, W] as input 0~255 in uint8
                t_start = time.time()
                image_ori_for_depthmodel = torch.tensor(viewpoint.image_ori).cuda().to(torch.uint8)
 
                # Both depth backends accept an RGB tensor and camera intrinsics.
                predictions = self.depth_model.infer(image_ori_for_depthmodel, self.intrinsics_ori) 

                if "mask" in predictions:

                    mask_pred_tensor = predictions["mask"].detach()
                    mask_pred = mask_pred_tensor.squeeze().cpu().numpy()

                else:

                    mask_pred_tensor = None
                    mask_pred = None
                    print("no mask")
                
                # Keep full-resolution depth for pose estimation.
                depth_pred_tensor = predictions["depth"].detach() # depth range: 0-255 in float

                depth_ori = depth_pred_tensor.squeeze().cpu().numpy() # shape: [H, W]
                # print(depth_ori.shape)
                viewpoint.depth_ori = depth_ori
                # depth_down = cv2.resize(depth_ori, (self.dataset.width, self.dataset.height))
                # Resize depth to the mapping resolution.
                depth_down_tensor = F.interpolate(depth_pred_tensor, size=(self.dataset.height, self.dataset.width), mode='bilinear', align_corners=False)
                viewpoint.depth = depth_down_tensor.squeeze(0)

                viewpoint.compute_grad_mask(self.config)
                t_depth = time.time()
                # Queue loop retrieval before estimating the current pose.
                self.classic_tracking.loop_kf(viewpoint.image_ori,  self.classic_tracking.kp_frame_id)
                t_loop = time.time()
                # Estimate motion and update the tracking state.
                #loop_flag = self.classic_tracking.update(viewpoint.image_ori, viewpoint.depth_ori, self.classic_tracking.kp_frame_id)
                loop_flag, s, loop_record = self.classic_tracking.update(viewpoint.image_ori, viewpoint.depth_ori, self.classic_tracking.kp_frame_id, viewpoint.uid, 
                                                           last_img=self.last_img, last_depth=self.last_depth, flow_mag=flow, sky_mask=mask_pred)
                t_tracking = time.time()
                print("Depth time: {:.3f}s, Loop Detection time: {:.3f}s, Tracking time: {:.3f}s".format(t_depth - t_start, t_loop - t_depth, t_tracking - t_loop))

                self.classic_tracking.kp_frame_id += 1
                
                self.classic_tracking.set_curr_keyframe(viewpoint.image_ori)

                if loop_flag:   # Apply loop-corrected poses to cameras and the map.
                    # new keyframe has not been added
                    poses_old, poses_update = self.classic_tracking.get_loop_update()   
                    poses_update = torch.tensor(poses_update)
                    poses_update = poses_update[:-1] # drop off the last pose (new keyframe)
                    poses_old = torch.tensor(poses_old)
                    poses_old = poses_old[:-1] # drop off the last pose (new keyframe)
                    update_viewpoints_from_poses(self.cameras, poses_update)    # Convert corrected C2W poses to camera W2C transforms.
                    self.request_loop_update(cur_frame_idx, poses_old, poses_update, self.kf_indices, loop_record)   
                
                # Save RAM space, we dont need original resolution for mapping
                viewpoint.image_ori = None
                viewpoint.depth_ori = None

                self.cameras[cur_frame_idx] = viewpoint
                self.keyframe_idx_list.append(cur_frame_idx)

                if self.reset:
                    self.initialize(cur_frame_idx, viewpoint)
                    self.current_window.append(cur_frame_idx)
                    cur_frame_idx += 1
                    continue

                self.initialized = self.initialized or (
                    len(self.current_window) == self.window_size
                )

                self.tracking(cur_frame_idx, viewpoint)

                self.current_window, removed = self.add_to_window(
                    cur_frame_idx,
                    self.current_window,
                )
                if self.monocular and not self.initialized and removed is not None:
                    self.reset = True
                    Log(
                        "Keyframes lacks sufficient overlap to initialize the map, resetting."
                    )
                    continue

                if self.viz and (cur_frame_idx - self.viz_frame_id_prev) > self.viz_frame_interval:
                    self.classic_tracking.viz_pose(self.classic_tracking.kp_frame_id)
                    self.viz_frame_id_prev = cur_frame_idx

                depth_map = self.add_new_keyframe(
                    cur_frame_idx,
                    init=False,
                )
                self.request_keyframe(
                    cur_frame_idx, viewpoint, self.current_window, depth_map
                )
                        
                cur_frame_idx += 1

                toc.record()
                torch.cuda.synchronize()
                if True:
                    duration = tic.elapsed_time(toc)
                    time.sleep(max(0.01, 1.0 / 3.0 - duration / 1000))

            else:
                data = self.frontend_queue.get()
                if data[0] == "sync_backend":
                    self.sync_backend(data)

                elif data[0] == "keyframe":
                    self.sync_backend(data)
                    self.requested_keyframe -= 1

                elif data[0] == "init":
                    self.sync_backend(data)
                    self.requested_init = False

                elif data[0] == "stop":
                    Log("Frontend Stopped.")
                    break

