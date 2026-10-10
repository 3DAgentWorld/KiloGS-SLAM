import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class TrackingDiagnoser:
    def __init__(self, save_dir, viz_depth_dist=False):
        self.save_dir = os.path.join(save_dir, "tracking_logs")
        os.makedirs(self.save_dir, exist_ok=True)
        
        
        self.viz_depth_dist = viz_depth_dist
        if self.viz_depth_dist:
            self.depth_log_dir = os.path.join(self.save_dir, "depth_debug")
            os.makedirs(self.depth_log_dir, exist_ok=True)
        
        
        self.history = {
            "frame_id": [],
            "mode": [],          # 0: E-Tracker, 1: PnP-Tracker
            "scale": [],         
            "inliers_2d": [],    
            "inliers_pnp": [],   
            "flow_mag": [],      
            "has_loop": []       
        }
        
        # Store sparse diagnostic events by frame ID.
        
        self.extra_events = {}

    def log(self, frame_id, mode, scale, inliers, flow_mag, has_loop):
        self.history["frame_id"].append(frame_id)
        
        mode_str = mode
        if isinstance(mode, int):
            if mode == 0: mode_str = "E"
            elif mode == 1: mode_str = "PnP"
            elif mode == 2: mode_str = "MASt3R"
            
        self.history["mode"].append(mode_str)
        self.history["scale"].append(scale)
        
        # Record inliers under the selected solver for plotting.
        
        is_pnp_like = (mode_str == "PnP" or mode_str == "MASt3R")
        
        self.history["inliers_2d"].append(inliers if mode_str == "E" else 0)
        self.history["inliers_pnp"].append(inliers if is_pnp_like else 0)
        
        self.history["flow_mag"].append(flow_mag)
        self.history["has_loop"].append(1 if has_loop else 0)
        
    
    def log_extra(self, frame_id, event_name, reason):
        """Record a diagnostic event and its reason for a frame."""
        msg = f"{event_name}: {reason}"
        
        # Append multiple events associated with the same frame.
        if frame_id in self.extra_events:
            self.extra_events[frame_id] += f" | {msg}"
        else:
            self.extra_events[frame_id] = msg

    def save_and_plot(self, inlier_thresh=120):
        
        df = pd.DataFrame(self.history)


        df['notes'] = df['frame_id'].map(self.extra_events)
        
        df['notes'] = df['notes'].fillna("")
        
        csv_path = os.path.join(self.save_dir, "tracking_stats.csv")
        df.to_csv(csv_path, index=False)
        
        
        # Only the active solver contributes to the per-frame inlier count.
        total_inliers = df['inliers_2d'] + df['inliers_pnp']
        bad_frames = df[total_inliers < inlier_thresh]['frame_id'].tolist()
        
        if len(bad_frames) > 0:
            print(f"\n[Diagnoser] ⚠️ Found {len(bad_frames)} frames with inliers < {inlier_thresh}:")
            print(f"[Diagnoser] Bad Frame IDs: {bad_frames}")
            
            
            with open(os.path.join(self.save_dir, "bad_frames_list.txt"), "w") as f:
                f.write(f"Threshold: {inlier_thresh}\n")
                f.write(f"Bad Frames: {bad_frames}\n")
        else:
            print(f"\n[Diagnoser] ✅ All frames have inliers >= {inlier_thresh}")

        
        fig, axs = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
        
        
        colors = ['blue' if m == 'E' else 'red' for m in df['mode']]
        axs[0].scatter(df['frame_id'], [1]*len(df), c=colors, s=10, label='Blue: E-Tracker, Red: PnP')
        axs[0].set_ylabel("Mode Trigger")
        axs[0].set_title("Tracking Mode (Red indicates fallback to PnP)")
        axs[0].legend(loc='upper right')
        
        
        axs[1].plot(df['frame_id'], df['scale'], color='green')
        axs[1].set_ylabel("Scale Factor")
        axs[1].set_title("Scale Consistency (Jumps indicate drift)")
        axs[1].grid(True)

        
        axs[2].plot(df['frame_id'], df['inliers_2d'], label='2D Inliers', alpha=0.5)
        axs[2].plot(df['frame_id'], df['inliers_pnp'], label='PnP Inliers', alpha=0.5)
        
        axs[2].axhline(y=inlier_thresh, color='red', linestyle='--', linewidth=2, label=f'Thresh={inlier_thresh}')
        
        axs[2].set_ylabel("Inlier Count")
        axs[2].set_title("Feature Matching Quality")
        axs[2].legend()

        
        axs[3].plot(df['frame_id'], df['flow_mag'], color='orange')
        axs[3].set_ylabel("Flow Mag")
        axs[3].set_title("Camera Motion Magnitude")

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "diagnosis_report.png"))
        plt.close()
        print(f"[Diagnoser] Report saved to {self.save_dir}")
        
    def visualize_scale_debug(self, frame_id, depth_raw, depth_proc, img_curr, img_prev, debug_data):
        """Plot depth, rendered references and trusted-region diagnostics."""
        if not self.viz_depth_dist:
            return

        render_rgb = debug_data.get("render_rgb")
        render_depth = debug_data.get("render_depth")
        mask_trust = debug_data.get("mask_trust")
        scatter_x = debug_data.get("scatter_x")
        scatter_y = debug_data.get("scatter_y")
        ransac_res = debug_data.get("ransac_line") # (s, bias)
        status = debug_data.get("status", "Unknown")
        feat_curr = debug_data.get("feat_curr") 

        
        fig, axs = plt.subplots(2, 4, figsize=(24, 10))
        axs = axs.flatten()
        
        cmap = 'magma'
        
        valid_raw = depth_raw[depth_raw > 0.1]
        vmin, vmax = (np.percentile(valid_raw, [1, 99]) if len(valid_raw)>0 else (0, 100))


        if img_curr.ndim == 3 and img_curr.shape[0] == 3:
            img_curr = np.transpose(img_curr, (1, 2, 0))
        
        if len(img_curr.shape) == 2:
            axs[0].imshow(img_curr, cmap='gray')
        else:
            axs[0].imshow(img_curr)
        title_str = f"Frame {frame_id} (Curr)"
        if feat_curr is not None and len(feat_curr) > 0:
            
            axs[0].scatter(feat_curr[:, 0], feat_curr[:, 1], s=3, c='cyan', alpha=0.6, label='Feat Curr')
            title_str += f"\nFeats: {len(feat_curr)}"
            
        
        axs[0].set_title(title_str)
        axs[0].axis('off')

        
        axs[1].imshow(depth_raw, cmap=cmap, vmin=vmin, vmax=vmax)
        axs[1].set_title("UniDepth (Raw)")
        axs[1].axis('off')

        
        if render_rgb is not None:
            if render_rgb.max() > 1.1: render_rgb = render_rgb / 255.0
            axs[2].imshow(np.clip(render_rgb, 0, 1))
            axs[2].set_title("Rendered RGB (Ref)")
        else:
            axs[2].text(0.5,0.5,"No Render", ha='center')
        axs[2].axis('off')

        
        if render_depth is not None:
            axs[3].imshow(render_depth, cmap=cmap, vmin=vmin, vmax=vmax)
            axs[3].set_title("Rendered Depth (Ref)")
        else:
            axs[3].text(0.5,0.5,"No Depth", ha='center')
        axs[3].axis('off')


        ratio_str = "N/A"
        num_pts = 0
        if mask_trust is not None and render_depth is not None:
            # Compute the median depth ratio within trusted regions.
            valid_raw_m = depth_raw[mask_trust]
            valid_map_m = render_depth[mask_trust]
            num_pts = np.sum(mask_trust)
            
            if valid_raw_m.size > 0 and valid_map_m.size > 0:
                med_raw = np.median(valid_raw_m)
                med_map = np.median(valid_map_m)
                
                ratio_val = med_map / med_raw if med_raw > 1e-3 else 0
                ratio_str = f"{ratio_val:.3f}"


        if mask_trust is not None:
            
            if len(img_prev.shape) == 2:
                axs[4].imshow(img_prev, cmap='gray')
            else:
                axs[4].imshow(img_prev)
            
            # Overlay trusted pixels without obscuring the rest of the image.
            
            masked_vis = np.ma.masked_where(~mask_trust, mask_trust)
            axs[4].imshow(masked_vis, cmap='Reds', alpha=0.6, vmin=0, vmax=1)
            
            valid_pct = num_pts / mask_trust.size * 100
            
            axs[4].set_title(f"Mask Overlay (N={num_pts} | {valid_pct:.1f}%)", color='red', fontweight='bold')
        else:
            axs[4].text(0.5, 0.5, "No Mask", ha='center')
            axs[4].axis('off')

        
        axs[5].imshow(depth_proc, cmap=cmap, vmin=vmin, vmax=vmax)
        axs[5].set_title("Final Preprocessed Depth")
        axs[5].axis('off')

        
        if scatter_x is not None and scatter_y is not None and len(scatter_x) > 0:
            axs[6].scatter(scatter_x, scatter_y, alpha=0.3, s=2, c='blue', label='Matches')
            
            
            if ransac_res is not None:
                s, bias = ransac_res
                x_line = np.array([scatter_x.min(), scatter_x.max()])
                y_line = s * x_line + bias
                axs[6].plot(x_line, y_line, 'r-', linewidth=2, label=f'y={s:.2f}x+{bias:.1f}')
                
                
                axs[6].set_title(f"Fit: Scale={s:.2f} | MedRatio={ratio_str}", fontsize=10, fontweight='bold')
            else:
                axs[6].set_title("Fit Failed")
                
            axs[6].set_xlabel("UniDepth Z")
            axs[6].set_ylabel("Map Z (Target)")
            axs[6].legend()
            axs[6].grid(True)
        else:
            axs[6].text(0.5, 0.5, f"No Points\nStatus: {status}", ha='center')

        
        valid_raw = depth_raw[depth_raw > 0.1].flatten()
        axs[7].hist(valid_raw, bins=50, color='teal', alpha=0.6, log=True, label='Raw')
        if render_depth is not None:
             valid_map = render_depth[render_depth > 0.1].flatten()
             if len(valid_map) > 0:
                 axs[7].hist(valid_map, bins=50, color='red', alpha=0.6, log=True, histtype='step', label='Map')
        axs[7].set_title("Depth Dist (Log)")
        axs[7].legend()

        plt.tight_layout()
        save_path = os.path.join(self.depth_log_dir, f"scale_debug_{frame_id:05d}.png")
        plt.savefig(save_path)
        plt.close()