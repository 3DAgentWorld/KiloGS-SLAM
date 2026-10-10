import random

import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from gaussian_splatting.gaussian_renderer import render, render_fastgs
from gaussian_splatting.utils.loss_utils import l1_loss, ssim, l2_loss, nearMean_map, image2canny
from utils.logging_utils import Log
from utils.multiprocessing_utils import clone_obj
from utils.pose_utils import update_pose, get_pose
from utils.slam_utils import get_loss_mapping, update_viewpoints_from_poses, update_point_cloud_in_batches
from utils.fastgs_utils import compute_photometric_loss, sampling_cameras, get_loss

from utils.anchor_utils import anchor_in_frustum
import numpy as np

import time

class BackEnd(mp.Process):
    def __init__(self, config, dataset, save_dir=None):
        super().__init__()
        self.dataset = dataset

        self.config = config
        self.gaussians = None
        self.pipeline_params = None
        self.opt_params = None
        self.background = None
        self.cameras_extent = None
        self.frontend_queue = None
        self.backend_queue = None
        self.live_mode = False
        self.save_dir = save_dir

        self.pause = False
        self.device = "cuda"
        self.dtype = torch.float32
        self.monocular = config["Training"]["monocular"]
        self.iteration_count = 0
        self.last_sent = 0
        self.occ_aware_visibility = {}
        self.viewpoints = {}
        self.current_window = []
        self.initialized = True
        self.keyframe_optimizers = None

        self.pose_opt = False
        self.adjust_anchor_map = False
        self.adjust_anchor_map = False

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
        
        self.trajectory = 255 + np.zeros((700, 700, 3), dtype=np.uint8)

        self.color_refinement_iter = config['Hierarchical']['color_refinement_iter']
        self.viz = config['SLAM']['viz']

    def set_hyperparams(self):
        self.save_results = self.config["Results"]["save_results"]

        self.init_itr_num = self.config["Training"]["init_itr_num"]
        self.init_gaussian_update = self.config["Training"]["init_gaussian_update"]
        self.init_gaussian_reset = self.config["Training"]["init_gaussian_reset"]
        self.init_gaussian_th = self.config["Training"]["init_gaussian_th"]
        self.init_gaussian_extent = (
            self.cameras_extent * self.config["Training"]["init_gaussian_extent"]
        )
        self.mapping_itr_num = self.config["Training"]["mapping_itr_num"]
        self.gaussian_update_every = self.config["Training"]["gaussian_update_every"]
        self.gaussian_update_offset = self.config["Training"]["gaussian_update_offset"]
        self.gaussian_th = self.config["Training"]["gaussian_th"]
        self.gaussian_extent = (
            self.cameras_extent * self.config["Training"]["gaussian_extent"]
        )
        self.gaussian_reset = self.config["Training"]["gaussian_reset"]
        self.size_threshold = self.config["Training"]["size_threshold"]
        self.window_size = self.config["Training"]["window_size"]
        self.single_thread = self.config["Training"]["single_thread"]

    def add_next_kf(self, frame_idx, viewpoint, init=False, scale=2.0, depth_map=None, rgb = None):
        Log(f'BACKEND: add_next_kf idx: {frame_idx}')
        render_pkg = None
        opt_mask = None
        if self.gaussians.get_anchor.shape[0] > 0:
            with torch.no_grad():
                opt_mask = torch.zeros(self.gaussians.get_anchor.shape[0], dtype=torch.bool, device='cuda')
                m = anchor_in_frustum(anchors=self.gaussians.get_anchor, 
                                            intrinsics=self.intrinsics, 
                                            pose=get_pose(viewpoint), 
                                            cam_center=viewpoint.camera_center,
                                            distance_lis = self.gaussians.distance_lis,
                                            levels=self.gaussians.get_level, 
                                            h=self.height, 
                                            w=self.width)
                opt_mask.bitwise_or_(m)
                render_pkg = render(
                    viewpoint,
                    self.gaussians,
                    self.pipeline_params,
                    self.background,
                    visible_mask=opt_mask
                )
        self.gaussians.extend_gaussian(
            viewpoint,
            anchor_index=frame_idx,
            init=init,
            depthmap=depth_map,
            rgb=rgb,
            render_pkg=render_pkg,
            frame_idx=frame_idx
        )
    def reset(self):
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.viewpoints = {}
        self.current_window = []
        self.initialized = True
        self.keyframe_optimizers = None
        # remove everything from the queues
        while not self.backend_queue.empty():
            self.backend_queue.get()

    def initialize_map(self, cur_frame_idx, viewpoint):
        for mapping_iteration in range(self.init_itr_num):
            self.iteration_count += 1
            opt_mask = torch.zeros(self.gaussians.get_anchor.shape[0], dtype=torch.bool, device='cuda')
            m = anchor_in_frustum(anchors=self.gaussians.get_anchor, 
                                        intrinsics=self.intrinsics, 
                                        pose=get_pose(viewpoint), 
                                        cam_center=viewpoint.camera_center,
                                        distance_lis = self.gaussians.distance_lis,
                                        levels=self.gaussians.get_level, 
                                        h=self.height, 
                                        w=self.width)
            opt_mask.bitwise_or_(m)
            
            render_pkg = render(
                viewpoint, 
                self.gaussians, 
                self.pipeline_params, 
                self.background, 
                visible_mask=opt_mask
            )
            (
                image,
            #    viewspace_point_tensor,
            #    visibility_filter,
            #    radii,
                depth,
                opacity,
            #    n_touched,
            ) = (
                render_pkg["render"],
            #    render_pkg["viewspace_points"],
            #    render_pkg["visibility_filter"],
            #    render_pkg["radii"],
                render_pkg["depth"],
                render_pkg["opacity"],
            #    render_pkg["n_touched"],
            )
            
            loss_init = get_loss_mapping(
                self.config, image, depth, viewpoint, opacity, initialization=True, save_dir=self.save_dir
            )
            loss_init.backward()

            with torch.no_grad():
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)

        Log("Initialized map")
        return render_pkg

    def compute_gaussian_score_fastgs(self, camlist, args, DENSIFY=False):
        """Compute multi-view consistency scores for Gaussians to guide densification.

        For each camera in `camlist` the function renders the scene and computes a
        photometric loss and a binary metric map of high-error pixels. It accumulates
        per-Gaussian counts of views that flagged the Gaussian and a weighted
        photometric score across views.

        Args:
            camlist (list): list of viewpoint camera objects to render from.
            gaussians: current Gaussian representation (model/state) used for rendering.
            pipe: rendering pipeline/context required by `render`.
            bg: background used for rendering.
            args: runtime config containing thresholds (e.g. `loss_thresh`).
            DENSIFY (bool): whether to compute and return the importance score
                used for densification. If False, only the pruning score is computed.

        Returns:
            importance_score (Tensor): per-Gaussian integer counts of how many views
                marked the Gaussian as high-error (floor-averaged across views).
                This output is only returned if `DENSIFY` is True.
            pruning_score (Tensor): normalized (0..1) per-Gaussian score used to
                prioritize densification (higher means worse reconstruction consistency).
        """

        full_metric_counts_anchor = None  # anchor-level counts
        full_metric_counts_offset = None  # offset-level counts
        full_metric_score = None         # anchor-level weighted score

        for view in range(len(camlist)):
            my_viewpoint_cam = camlist[view]

            render_image = render_fastgs(my_viewpoint_cam, self.gaussians, self.pipeline_params, self.background)["render"]
            photometric_loss = compute_photometric_loss(my_viewpoint_cam, render_image)

            gt_image = my_viewpoint_cam.original_image.cuda()
            get_flag = True
            pixel_l1 = get_loss(render_image, gt_image)

            metric_map = (pixel_l1 > args["loss_thresh"]).int()
            render_pkg = render_fastgs(my_viewpoint_cam, self.gaussians, self.pipeline_params, self.background, get_flag = get_flag, metric_map = metric_map)

            accum_loss_counts = render_pkg["accum_metric_counts"]

            # Offset-level counts (no aggregation) kept for densify/importance.
            if full_metric_counts_offset is None:
                full_metric_counts_offset = accum_loss_counts.clone()
            else:
                if full_metric_counts_offset.numel() != accum_loss_counts.numel():
                    full_metric_counts_offset = torch.zeros_like(accum_loss_counts)
                full_metric_counts_offset += accum_loss_counts

            # Anchor-level counts: reshape/sum over offsets, truncate to anchor length.
            n_offsets = self.gaussians.n_offsets
            num_offsets = accum_loss_counts.numel()
            if num_offsets == 0:
                continue
            num_anchors = min(num_offsets // n_offsets, self.gaussians.get_anchor.shape[0])
            counts_anchor = accum_loss_counts[: num_anchors * n_offsets]
            counts_anchor = counts_anchor.view(num_anchors, n_offsets).sum(dim=1)

            if full_metric_counts_anchor is None:
                full_metric_counts_anchor = counts_anchor.clone()
            else:
                if full_metric_counts_anchor.numel() != counts_anchor.numel():
                    full_metric_counts_anchor = torch.zeros_like(counts_anchor)
                    if full_metric_score is not None:
                        full_metric_score = torch.zeros_like(counts_anchor, dtype=full_metric_score.dtype)
                full_metric_counts_anchor += counts_anchor

            contrib = photometric_loss * counts_anchor
            if full_metric_score is None:
                full_metric_score = contrib.clone()
            else:
                if full_metric_score.numel() != contrib.numel():
                    full_metric_score = torch.zeros_like(contrib)
                full_metric_score += contrib

        # Avoid division by zero if all scores are identical
        if full_metric_score is None:
            pruning_score = torch.zeros((self.gaussians.get_anchor.shape[0],), device="cuda")
        else:
            score_min = torch.min(full_metric_score)
            score_max = torch.max(full_metric_score)
            denom = score_max - score_min
            if denom == 0:
                pruning_score = torch.zeros_like(full_metric_score)
            else:
                pruning_score = (full_metric_score - score_min) / denom

        # Importance scores align with individual Gaussian offsets.
        if DENSIFY and full_metric_counts_offset is not None:
            importance_score = torch.div(full_metric_counts_offset, len(camlist), rounding_mode='floor')
        else:
            importance_score = None

        # Pruning scores align with anchors.
        return importance_score, pruning_score

    def compute_gaussian_score_fastgs_masked(self, camlist, args, DENSIFY=False):
        """
        Accumulate scores over the union of sampled camera frusta.
        Importance is the floor-averaged offset error count; pruning uses
        log scores normalized by their maximum (quantile 1.0).
        """
        # Render only anchors visible in at least one sampled camera.
        num_anchors = self.gaussians.get_anchor.shape[0]
        union_mask = torch.zeros(num_anchors, dtype=torch.bool, device='cuda')
        
        for cam in camlist:
            m = anchor_in_frustum(anchors=self.gaussians.get_anchor, 
                                  intrinsics=self.intrinsics, 
                                  pose=get_pose(cam), 
                                  cam_center=cam.camera_center,
                                  distance_lis=self.gaussians.distance_lis,
                                  levels=self.gaussians.get_level,
                                  h=self.height, w=self.width)
            union_mask.bitwise_or_(m)
            
        if union_mask.sum() == 0:
            return None, None

        
        full_metric_counts_offset_global = torch.zeros(num_anchors * self.gaussians.n_offsets, device='cuda')
        full_metric_score_global = torch.zeros(num_anchors, device='cuda')
        union_mask_offset = union_mask.unsqueeze(1).repeat(1, self.gaussians.n_offsets).view(-1)

        
        for view in range(len(camlist)):
            my_viewpoint_cam = camlist[view]


            gt_image = my_viewpoint_cam.original_image.cuda()
            
            # Render color to construct the photometric error mask.
            render_pkg = render_fastgs(my_viewpoint_cam, self.gaussians, self.pipeline_params, self.background, visible_mask=union_mask)
            render_image = render_pkg["render"]
            
            pixel_l1 = get_loss(render_image, gt_image) 
            metric_map = (pixel_l1 > args["loss_thresh"]).int()
            photometric_loss = compute_photometric_loss(my_viewpoint_cam, render_image)
            
            del render_pkg

            # Render again to accumulate per-offset error counts.
            render_pkg_metric = render_fastgs(my_viewpoint_cam, self.gaussians, self.pipeline_params, self.background, 
                                              visible_mask=union_mask, get_flag=True, metric_map=metric_map)
            
            accum_loss_counts = render_pkg_metric["accum_metric_counts"]
            
            
            full_metric_counts_offset_global[union_mask_offset] += accum_loss_counts

            num_visible_anchors = int(union_mask.sum().item())
            counts_anchor_local = accum_loss_counts.view(num_visible_anchors, self.gaussians.n_offsets).sum(dim=1)
            contrib = photometric_loss * counts_anchor_local
            full_metric_score_global[union_mask] += contrib
            
            del render_pkg_metric, accum_loss_counts, render_image, pixel_l1, metric_map

        # --- 4. Pruning Score (Log + Robust Norm) ---
        if full_metric_score_global.sum() == 0:
             pruning_score = torch.zeros(num_anchors, device="cuda")
        else:
            valid_scores = full_metric_score_global[union_mask]
            
            # Normalize log scores by their observed maximum (quantile 1.0).
            valid_scores_log = torch.log1p(valid_scores)
            score_min = valid_scores_log.min()

            score_q99 = torch.quantile(valid_scores_log, 1)
            valid_scores_clipped = torch.clamp(valid_scores_log, max=score_q99)
            denom = score_q99 - score_min
            
            pruning_score = torch.zeros_like(full_metric_score_global)
            if denom > 1e-6:
                pruning_score[union_mask] = (valid_scores_clipped - score_min) / denom

        
        if DENSIFY:
            # Floor-average offset counts across the sampled cameras.
            
            importance_score = torch.div(full_metric_counts_offset_global, len(camlist), rounding_mode='floor')
        else:
            importance_score = None

        return importance_score, pruning_score
    
    def map(self, current_window, prune=False, iters=1, time_return = False, random = True):
        if len(current_window) == 0:
            return

        viewpoint_stack = [self.viewpoints[kf_idx] for kf_idx in current_window]
        random_viewpoint_stack = []
        frames_to_optimize = self.config["Training"]["pose_window"]

        current_window_set = set(current_window)
        for cam_idx, viewpoint in self.viewpoints.items():
            if cam_idx in current_window_set:
                continue
            random_viewpoint_stack.append(viewpoint)
        
        frustum_time = 0

        for _ in range(iters):
            self.iteration_count += 1
            self.last_sent += 1

            loss_mapping = 0
            viewspace_point_tensor_acm = []
            visibility_filter_acm = []
            radii_acm = []
            n_touched_acm = []
            offset_selection_mask_acm = []
            voxel_visible_mask_acm = []
            neural_opacity_acm = []

            keyframes_opt = []

            for cam_idx in range(len(current_window)):
                viewpoint = viewpoint_stack[cam_idx]
                keyframes_opt.append(viewpoint)

                opt_mask = torch.zeros(self.gaussians.get_anchor.shape[0], dtype=torch.bool, device='cuda')
                m = anchor_in_frustum(anchors=self.gaussians.get_anchor, 
                                            intrinsics=self.intrinsics, 
                                            pose=get_pose(viewpoint), 
                                            cam_center=viewpoint.camera_center,
                                            distance_lis = self.gaussians.distance_lis,
                                            levels=self.gaussians.get_level, 
                                            h=self.height, 
                                            w=self.width)
                opt_mask.bitwise_or_(m)

                
                render_pkg = render(
                    viewpoint, 
                    self.gaussians, 
                    self.pipeline_params, 
                    self.background, 
                    visible_mask=opt_mask
                )
                (
                    image,
                    #viewspace_point_tensor,
                    #visibility_filter,
                    #radii,
                    depth,
                    opacity,
                    #n_touched,
                    # neural_opacity,
                    # offset_selection_mask
                ) = (
                    render_pkg["render"],
                    #render_pkg["viewspace_points"],
                    #render_pkg["visibility_filter"],
                    #render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                    #render_pkg["n_touched"],
                    # render_pkg["neural_opacity"],
                    # render_pkg["selection_mask"],
                )

                loss_mapping += get_loss_mapping(
                    self.config, image, depth, viewpoint, opacity, initialization=False, save_dir=self.save_dir
                )

            if random:
                for cam_idx in torch.randperm(len(random_viewpoint_stack))[:2]:
                    viewpoint = random_viewpoint_stack[cam_idx]
                    opt_mask = torch.zeros(self.gaussians.get_anchor.shape[0], dtype=torch.bool, device='cuda')

                    m = anchor_in_frustum(anchors=self.gaussians.get_anchor, 
                                                intrinsics=self.intrinsics, 
                                                pose=get_pose(viewpoint), 
                                                cam_center=viewpoint.camera_center,
                                                distance_lis = self.gaussians.distance_lis,
                                                levels=self.gaussians.get_level, 
                                                h=self.height, 
                                                w=self.width)
                    opt_mask.bitwise_or_(m)
                    
                    render_pkg = render(
                        viewpoint, 
                        self.gaussians, 
                        self.pipeline_params, 
                        self.background, 
                        visible_mask=opt_mask
                    )
                    (
                        image,
                        #viewspace_point_tensor,
                        #visibility_filter,
                        #radii,
                        depth,
                        opacity,
                        #n_touched,
                        # neural_opacity,
                        # offset_selection_mask
                    ) = (
                        render_pkg["render"],
                        #render_pkg["viewspace_points"],
                        #render_pkg["visibility_filter"],
                        #render_pkg["radii"],
                        render_pkg["depth"],
                        render_pkg["opacity"],
                        #render_pkg["n_touched"],
                        # render_pkg["neural_opacity"],
                        # render_pkg["selection_mask"],
                    )

                    loss_mapping += get_loss_mapping(
                        self.config, image, depth, viewpoint, opacity, initialization=False, save_dir=self.save_dir
                    )

            scaling = self.gaussians.get_scaling
            isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
            loss_mapping += 10 * isotropic_loss.mean()

            loss_mapping.backward()

            gaussian_split = False
            with torch.no_grad():

                if self.adjust_anchor_map:
                    for idx in range(len(viewspace_point_tensor_acm)):
                        self.gaussians.training_statis(viewspace_point_tensor_acm[idx], neural_opacity_acm[idx], visibility_filter_acm[idx], offset_selection_mask_acm[idx], voxel_visible_mask_acm[idx])
                    update_gaussian = (
                        self.iteration_count % self.gaussian_update_every
                        == self.gaussian_update_offset
                    )

                    if update_gaussian:
                        camlist = sampling_cameras(self.viewpoints.copy())
                        importance_score, pruning_score = self.compute_gaussian_score_fastgs(camlist, self.config["Training"], DENSIFY = True)
                        print("Updating Gaussians...")
                        self.gaussians.adjust_anchor_fastgs(pruning_score = pruning_score)

                self.gaussians.optimizer.step()

                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(self.iteration_count)

                if self.pose_opt:
                    self.keyframe_optimizers.step()
                    self.keyframe_optimizers.zero_grad(set_to_none=True)

                # Pose update
                for cam_idx in range(min(frames_to_optimize, len(current_window))):
                    viewpoint = viewpoint_stack[cam_idx]
                    if viewpoint.uid == 0:
                        continue
                    update_pose(viewpoint)
        
        if time_return:
            return gaussian_split, frustum_time
        else:
            return gaussian_split

    def color_refinement(self):
        Log("Starting global color refinement with segmented FastGS update")

        # Partition keyframes into chunks for multiview scoring.
        all_kf_indices = sorted(list(self.viewpoints.keys()))
        num_total_frames = len(all_kf_indices)
        chunk_size = 100
        
        chunks = []
        i = 0
        while i < num_total_frames:
            end = i + chunk_size
            if num_total_frames - end < chunk_size: end = num_total_frames
            chunks.append(all_kf_indices[i:end])
            i = end
            if i >= num_total_frames: break
        
        Log(f"Keyframes split into {len(chunks)} chunks.")

        
        iteration_total = self.color_refinement_iter * num_total_frames
        iteration_stop = 20 * num_total_frames  
        
        
        iter_start_stat   = 2 * num_total_frames
        iter_start_update = 5 * num_total_frames
        iter_stop_densify = 15 * num_total_frames
        update_interval   = 5 * num_total_frames 

        for iteration in tqdm(range(1, iteration_total + 1)):
            self.iteration_count += 1
            
            # Sample globally for color optimization.
            viewpoint_cam_idx = all_kf_indices[random.randint(0, num_total_frames - 1)]
            viewpoint_cam = self.viewpoints[viewpoint_cam_idx]
            
            opt_mask = torch.zeros(self.gaussians.get_anchor.shape[0], dtype=torch.bool, device='cuda')
            m = anchor_in_frustum(anchors=self.gaussians.get_anchor, intrinsics=self.intrinsics, pose=get_pose(viewpoint_cam), 
                                  cam_center=viewpoint_cam.camera_center, distance_lis=self.gaussians.distance_lis,
                                  levels=self.gaussians.get_level, h=self.height, w=self.width)
            opt_mask.bitwise_or_(m)
            
            render_pkg = render(viewpoint_cam, self.gaussians, self.pipeline_params, self.background, visible_mask=opt_mask)
            
            # Loss Backward ...
            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(render_pkg["render"], gt_image)
            ssim_loss = (1.0 - ssim(render_pkg["render"], gt_image))
            scaling_reg = render_pkg["scaling"].prod(dim=1).mean()
            loss = (1.0 - self.opt_params.lambda_dssim) * Ll1 + self.opt_params.lambda_dssim * ssim_loss + 0.05 * scaling_reg
            loss.backward()

            # Collect gradients and periodically update Gaussian structure.
            with torch.no_grad():
                if iteration > iter_start_stat:
                    self.gaussians.training_statis(
                        render_pkg["viewspace_points"], render_pkg["neural_opacity"], 
                        render_pkg["visibility_filter"], render_pkg["selection_mask"], opt_mask
                    )

                is_update_step = (iteration % update_interval == 0) and (iteration >= iter_start_update) and (iteration < iteration_stop)
                
                if is_update_step:
                    allow_densify = iteration < iter_stop_densify
                    Log(f"\n[Iter {iteration}] Update. Densify: {allow_densify}")
                    
                    
                    num_anchors = self.gaussians.get_anchor.shape[0]
                    
                    # Combine densification decisions across chunks with logical OR.
                    global_densify_mask = torch.zeros(num_anchors * self.gaussians.n_offsets, dtype=torch.bool, device='cuda')
                    global_pruning_max = torch.zeros(num_anchors, device='cuda')
                    
                    
                    for chunk_idx, chunk_indices in enumerate(chunks):
                        
                        n_sample = max(5, len(chunk_indices) // 5) 
                        sampled_indices = random.sample(chunk_indices, n_sample)
                        camlist = [self.viewpoints[uid] for uid in sampled_indices]
                        
                        
                        imp, prune = self.compute_gaussian_score_fastgs_masked(
                            camlist, self.config["Training"], DENSIFY=allow_densify
                        )
                        
                        
                        if imp is not None:
                            # Compare importance against half the sampled camera count.
                            chunk_threshold = max(1, len(camlist) // 2)               
                            
                            chunk_densify_mask = (imp.float() > chunk_threshold)      
                            
                            global_densify_mask = torch.logical_or(global_densify_mask, chunk_densify_mask)   
                        
                        
                        # if imp is not None:
                        
                        #     chunk_threshold = 0.5 
                        #     chunk_densify_mask = (imp.float() > chunk_threshold)
                        #     global_densify_mask = torch.logical_or(global_densify_mask, chunk_densify_mask)
                            
                        # Keep the maximum pruning score across chunks.
                        if prune is not None:
                            global_pruning_max = torch.maximum(global_pruning_max, prune)
                            
                        del imp, prune
                        torch.cuda.empty_cache()


                    self.gaussians.adjust_anchor_hybrid(
                        check_interval=update_interval,
                        grad_threshold=self.opt_params.densify_grad_threshold,
                        min_opacity=self.opt_params.min_opacity,
                        fastgs_densify_mask=global_densify_mask, 
                        pruning_score=global_pruning_max,
                        enable_densify=allow_densify,
                        enable_prune=True
                    )
                    
                    del global_densify_mask, global_pruning_max
                    torch.cuda.empty_cache()

                # Optimizer Step
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(self.iteration_count)

        Log("Map refinement done")


    def push_to_frontend(self, tag=None, send_gaussians=False, img = None, depth = None):
        self.last_sent = 0
        keyframes = []
        for kf_idx in self.current_window:
            kf = self.viewpoints[kf_idx]
            keyframes.append((kf_idx, kf.R.clone(), kf.T.clone()))
        if tag is None:
            tag = "sync_backend"

        gauss_payload = clone_obj(self.gaussians) if send_gaussians else None

        if img is not None:
            msg = [tag, gauss_payload, None, keyframes, img.detach(), depth.detach()]
        else:
            msg = [tag, gauss_payload, None, keyframes, None, None]
        self.frontend_queue.put(msg)

    def run(self):
        import time
        while True:
            if self.backend_queue.empty():
                if self.pause:
                    time.sleep(0.01)
                    continue
                if len(self.current_window) == 0:
                    time.sleep(0.01)
                    continue

                if self.single_thread:
                    time.sleep(0.01)
                    continue
                if self.last_sent >= 10:
                    self.push_to_frontend()
            else:
                data = self.backend_queue.get()
                if data[0] == "stop":
                    break

                elif data[0] == "pause":
                    self.pause = True
                elif data[0] == "unpause":
                    self.pause = False
                elif data[0] == "color_refinement":
                    self.color_refinement()
                    #self.push_to_frontend()
                    self.push_to_frontend(tag="color_refinement_done", send_gaussians=True)
                elif data[0] == "export_gaussians":
                    self.push_to_frontend(tag="export_gaussians", send_gaussians=True)
                elif data[0] == "init":
                    cur_frame_idx = data[1]
                    viewpoint = data[2]
                    depth_map = data[3]
                    Log("Resetting the system")
                    self.reset()

                    self.viewpoints[cur_frame_idx] = viewpoint
                    self.orig_viewpoint = viewpoint
                    self.add_next_kf(
                        cur_frame_idx, viewpoint, depth_map=depth_map, init=True, rgb=viewpoint.original_image
                    )
                    self.initialize_map(cur_frame_idx, viewpoint)
                    viewpoint = self.viewpoints[cur_frame_idx]
                    opt_mask = torch.zeros(self.gaussians.get_anchor.shape[0], dtype=torch.bool, device='cuda')
                    m = anchor_in_frustum(anchors=self.gaussians.get_anchor, 
                                                intrinsics=self.intrinsics, 
                                                pose=get_pose(viewpoint), 
                                                cam_center=viewpoint.camera_center,
                                                distance_lis = self.gaussians.distance_lis,
                                                levels=self.gaussians.get_level, 
                                                h=self.height, 
                                                w=self.width)
                    opt_mask.bitwise_or_(m)
                    
                    render_pkg = render(
                        viewpoint, 
                        self.gaussians, 
                        self.pipeline_params, 
                        self.background, 
                        visible_mask=opt_mask
                    )
                    (
                        image,
                        # viewspace_point_tensor,
                        # visibility_filter,
                        # radii,
                        depth,
                        # opacity,
                        # n_touched,
                    ) = (
                        render_pkg["render"],
                        # render_pkg["viewspace_points"],
                        # render_pkg["visibility_filter"],
                        # render_pkg["radii"],
                        render_pkg["depth"],
                        # render_pkg["opacity"],
                        # render_pkg["n_touched"],
                    )
                    self.push_to_frontend("init", img = image, depth = depth)

                elif data[0] == "loop_update":
                    cur_frame_idx = data[1]
                    poses_old = data[2]
                    poses_update = data[3]
                    kf_indices = data[4]
                    loop_record = data[5]
                    (i,j) = loop_record
                    # Log(f"Loop closure update received for frames {i} and {j}")
                    # Log(f"Loop closure update received for frames {kf_indices[i]} and {kf_indices[j]}")
                    
                    # print(torch.topk((poses_update[:, :3, 3] - poses_old[:, :3, 3]).norm(dim=1), 50))

                    update_viewpoints_from_poses(self.viewpoints, torch.tensor(poses_update))
                    updated_pcd = update_point_cloud_in_batches(self.gaussians.get_anchor,
                                                                poses_old, 
                                                                poses_update, 
                                                                self.gaussians.get_anchor_index,
                                                                kf_indices)
                    self.gaussians.update_anchor_loop(updated_pcd.cuda())
                    
                    # =========================================================
                    
                    # =========================================================
                    # try:
                    
                    
                    #     idx_start = max(0, j - 2)
                    #     idx_end = min(len(kf_indices) - 1, j + 2)


                    #     loop_neighborhood_ids = []
                    #     for k in range(idx_start, idx_end + 1):
                    #         val = kf_indices[k]
                    
                    #         kf_id = val.item() if hasattr(val, 'item') else val
                    #         loop_neighborhood_ids.append(kf_id)


                    #     temp_window = list(set(self.current_window + loop_neighborhood_ids))
                        
                    #     Log(f"[Loop Refine] Optimizing {len(temp_window)} frames (Current + Loop Neighborhood {loop_neighborhood_ids})")


                    #     _, _ = self.map(temp_window, iters=10, time_return=True, random=False)
                        
                    # except Exception as e:
                    #     Log(f"[Loop Refine Error] Failed to refine: {e}")

                elif data[0] == "keyframe":
                    begin = time.time()
                    cur_frame_idx = data[1]
                    viewpoint = data[2]
                    current_window = data[3]
                    depth_map = data[4]

                    self.viewpoints[cur_frame_idx] = viewpoint
                    self.current_window = current_window
                    
                    t_start = time.time()
                    
                    self.add_next_kf(cur_frame_idx, viewpoint, depth_map=depth_map, rgb=viewpoint.original_image)
                    
                    t_init = time.time()

                    opt_params = []
                    frames_to_optimize = self.config["Training"]["pose_window"]
                    kf_iter_num = self.config["Training"]["kf_iter_num"]
                    iter_per_kf = kf_iter_num
                    print(iter_per_kf, "kf_iter_num is:", kf_iter_num)

                    if not self.initialized:
                        print(self.initialized)
                        #print("Current window is:",self.current_window)
                        #print("len(self.current_window)=", len(self.current_window))
                        #print("window_size=", self.config["Training"]["window_size"])
                        if (
                            len(self.current_window)
                            == self.config["Training"]["window_size"]
                        ):
                            frames_to_optimize = (
                                self.config["Training"]["window_size"] - 1
                            )
                            iter_per_kf = 50 if self.live_mode else 300
                            Log("Performing initial BA for initialization")
                        else:
                            iter_per_kf = self.mapping_itr_num

                    pose_window = self.current_window[len(self.current_window) // 2:]

                    if len(pose_window) == 0 or (pose_window == [0]):
                        self.pose_opt = False
                    else:
                        self.pose_opt = True

                    for cam_idx in range(len(pose_window)):
                        if pose_window[cam_idx] == 0:
                            continue
                        viewpoint = self.viewpoints[pose_window[cam_idx]]
                        if cam_idx < frames_to_optimize:
                            opt_params.append(
                                {
                                    "params": [viewpoint.cam_rot_delta],
                                    "lr": self.config["Training"]["lr"]["cam_rot_delta"]
                                    * 0.5,
                                    "name": "rot_{}".format(viewpoint.uid),
                                }
                            )
                            opt_params.append(
                                {
                                    "params": [viewpoint.cam_trans_delta],
                                    "lr": self.config["Training"]["lr"][
                                        "cam_trans_delta"
                                    ]
                                    * 0.5,
                                    "name": "trans_{}".format(viewpoint.uid),
                                }
                            )
                        opt_params.append(
                            {
                                "params": [viewpoint.exposure_a],
                                "lr": 0.01,
                                "name": "exposure_a_{}".format(viewpoint.uid),
                            }
                        )
                        opt_params.append(
                            {
                                "params": [viewpoint.exposure_b],
                                "lr": 0.01,
                                "name": "exposure_b_{}".format(viewpoint.uid),
                            }
                        )
                    if self.pose_opt:
                        self.keyframe_optimizers = torch.optim.Adam(opt_params)


                    # print("iter number is:", iter_per_kf)
                    print(self.current_window, "iter num is:", iter_per_kf)
                    _, time_return = self.map(self.current_window, iters=iter_per_kf, time_return = True)
                    t_end = time.time()
                    print("init time:", t_init - t_start, "mapping time:", t_end - t_init)

                    viewpoint = self.viewpoints[cur_frame_idx]
                    opt_mask = torch.zeros(self.gaussians.get_anchor.shape[0], dtype=torch.bool, device='cuda')
                    m = anchor_in_frustum(anchors=self.gaussians.get_anchor, 
                                                intrinsics=self.intrinsics, 
                                                pose=get_pose(viewpoint), 
                                                cam_center=viewpoint.camera_center,
                                                distance_lis = self.gaussians.distance_lis,
                                                levels=self.gaussians.get_level, 
                                                h=self.height, 
                                                w=self.width)
                    opt_mask.bitwise_or_(m)
                    
                    render_pkg = render(
                        viewpoint, 
                        self.gaussians, 
                        self.pipeline_params, 
                        self.background, 
                        visible_mask=opt_mask
                    )
                    (
                        image,
                        # viewspace_point_tensor,
                        # visibility_filter,
                        # radii,
                        depth,
                        # opacity,
                        # n_touched,
                    ) = (
                        render_pkg["render"],
                        # render_pkg["viewspace_points"],
                        # render_pkg["visibility_filter"],
                        # render_pkg["radii"],
                        render_pkg["depth"],
                        # render_pkg["opacity"],
                        # render_pkg["n_touched"],
                    )

                    self.push_to_frontend("keyframe",img = image, depth = depth)
                else:
                    raise Exception("Unprocessed data", data)
        while not self.backend_queue.empty():
            self.backend_queue.get()
        while not self.frontend_queue.empty():
            self.frontend_queue.get()
        return
