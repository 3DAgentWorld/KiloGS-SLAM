import os
import sys
import time
from argparse import ArgumentParser, BooleanOptionalAction
from datetime import datetime
import gc
import signal
import traceback

import torch
import torch.multiprocessing as mp
import yaml
from munch import munchify

from gaussian_splatting.utils.system_utils import mkdir_p
from utils.config_utils import load_config
from utils.dataset import load_dataset
from utils.logging_utils import Log
from utils.multiprocessing_utils import FakeQueue
from utils.slam_frontend import FrontEnd

import warnings
warnings.filterwarnings("ignore")

class SLAM:
    def __init__(self, config, save_dir=None):
        if config.get("SLAM", {}).get("mode", "full") == "frontend":
            from utils.frontend_only import run_frontend_only
            self.eval_rendering = False
            self.color_refine_time = 0.0
            self.psnr_before = self.psnr_after = None
            self.frontend_stats = run_frontend_only(config, save_dir)
            self.slam_time = self.frontend_stats["total_seconds"]
            return

        # Mapping dependencies are loaded only in full mode.
        from gaussian_splatting.scene.scaffold_model import GaussianModel
        from gui import gui_utils, slam_gui
        from utils.eval_utils import eval_rendering, eval_ate
        from utils.slam_backend import BackEnd

        # Measure core SLAM time separately from evaluation and color refinement.
        slam_core_start = time.time()

        # Stats holders
        self.slam_time = 0.0
        self.color_refine_time = 0.0
        self.psnr_before = None
        self.psnr_after = None

        self.config = config
        self.save_dir = save_dir
        model_params = munchify(config["model_params"])
        opt_params = munchify(config["opt_params"])
        pipeline_params = munchify(config["pipeline_params"])
        self.model_params, self.opt_params, self.pipeline_params = (
            model_params,
            opt_params,
            pipeline_params,
        )

        self.live_mode = self.config["Dataset"]["type"] == "realsense"
        self.monocular = self.config["Dataset"]["sensor_type"] == "monocular"

        self.use_spherical_harmonics = self.config["Training"]["spherical_harmonics"]
        self.use_gui = self.config["Results"]["use_gui"]
        if self.live_mode:
            self.use_gui = True
        self.eval_rendering = self.config["Results"]["eval_rendering"]

        model_params.sh_degree = 3 if self.use_spherical_harmonics else 0
        self.dataset = load_dataset(
            model_params, model_params.source_path, config=config
        )
        if config["SLAM"].get("max_frames") is not None:
            self.dataset.num_imgs = min(len(self.dataset), config["SLAM"]["max_frames"])
        intrinsics = torch.tensor([self.dataset.fx, 
                                   self.dataset.fy, 
                                   self.dataset.cx, 
                                   self.dataset.cy], device='cuda')
        

        # ===== Scaffold GS ===== #
        self.feat_dim = 32
        self.n_offsets = self.config['Hierarchical']['n_offsets'] # 16
        self.voxel_size =  0.005 # if voxel_size<=0, using 1nn dist

        self.voxel_size_lis = self.config['Hierarchical']['voxel_size_lis']
        self.distance_lis = self.config['Hierarchical']['distance_lis']

        self.max_level = len(self.voxel_size_lis) - 1

        self.update_depth = 3
        self.update_init_factor = 16
        self.update_hierachy_factor = 4

        self.use_feat_bank = False
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False

        self.appearance_dim = self.config['Hierarchical']['appearance_dim']
        self.lowpoly = False
        self.ds = 1
        self.ratio = 1 # sampling the input point cloud
        self.undistorted = False 
        
        # In the Bungeenerf dataset, we propose to set the following three parameters to True,
        # Because there are enough dist variations.
        self.add_opacity_dist = False
        self.add_cov_dist = False
        self.add_color_dist = False

        self.gaussians = GaussianModel(self.feat_dim, 
                                    self.n_offsets, 
                                    self.voxel_size_lis, 
                                    self.distance_lis,
                                    self.update_depth, 
                                    self.update_init_factor, 
                                    self.update_hierachy_factor, 
                                    self.use_feat_bank, 
                                    self.appearance_dim, 
                                    self.ratio, 
                                    self.add_opacity_dist, 
                                    self.add_cov_dist, 
                                    self.add_color_dist,
                                    config = self.config, 
                                    intrinsics = intrinsics)
        self.gaussians.init_lr(6.0)
        self.gaussians.save_dir = save_dir

        # ===== Scaffold GS ===== #

        bg_color = [1, 1, 1]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        frontend_queue = mp.Queue()
        backend_queue = mp.Queue()

        q_main2vis = mp.Queue() if self.use_gui else FakeQueue()
        q_vis2main = mp.Queue() if self.use_gui else FakeQueue()
        self.frontend_queue, self.backend_queue = frontend_queue, backend_queue
        self.q_main2vis, self.q_vis2main = q_main2vis, q_vis2main
        self.gui_process = None

        self.config["Results"]["save_dir"] = save_dir
        self.config["Training"]["monocular"] = self.monocular

        ctx = mp.get_context("spawn")
        self.stop_event = ctx.Event()
        self.frontend = FrontEnd(self.config, self.dataset, save_dir = self.save_dir)
        self.backend = BackEnd(self.config, self.dataset, save_dir = self.save_dir)
        self.backend.stop_event = self.stop_event

        self.frontend.background = self.background
        self.frontend.pipeline_params = self.pipeline_params
        self.frontend.frontend_queue = frontend_queue
        self.frontend.backend_queue = backend_queue
        self.frontend.q_main2vis = q_main2vis
        self.frontend.q_vis2main = q_vis2main
        self.frontend.set_hyperparams()

        self.backend.gaussians = self.gaussians
        self.backend.background = self.background
        self.backend.cameras_extent = 6.0
        self.backend.pipeline_params = self.pipeline_params
        self.backend.opt_params = self.opt_params
        self.backend.frontend_queue = frontend_queue
        self.backend.backend_queue = backend_queue
        self.backend.live_mode = self.live_mode

        self.backend.set_hyperparams()

        self.params_gui = gui_utils.ParamsGUI(
            pipe=self.pipeline_params,
            background=self.background,
            gaussians=self.gaussians,
            q_main2vis=q_main2vis,
            q_vis2main=q_vis2main,
        )
        
        self.backend_process = mp.Process(target=self.backend.run)
        if self.use_gui:
            gui_process = mp.Process(target=slam_gui.run, args=(self.params_gui,))
            self.gui_process = gui_process
            gui_process.start()
            time.sleep(5)

        self.backend_process.start()
        self.frontend.run()
        backend_queue.put(["pause"])
        # Core time excludes rendering evaluation and color refinement.
        slam_core_end = time.time()
        self.slam_time = slam_core_end - slam_core_start
        Log(f"[INFO] SLAM core time (before color refine): {self.slam_time:.2f} s")
        # empty the frontend queue

        if self.eval_rendering:

            kf_indices = self.frontend.kf_indices
            # Request a gaussians snapshot from backend for evaluation
            backend_queue.put(["export_gaussians"])
            while True:
                if frontend_queue.empty():
                    time.sleep(0.01)
                    continue
                data = frontend_queue.get()
                if data[0] == "export_gaussians":
                    self.gaussians = data[1]
                    break
            
            # # evaluate ATE
            # ATE = eval_ate(
            #     self.frontend.cameras,
            #     kf_indices,
            #     self.save_dir,
            #     0,
            #     final=True,
            #     monocular=self.monocular,
            # )
            
            # 1) Evaluate rendering & PSNR BEFORE color refinement
            rendering_result_before = eval_rendering(
                self.frontend.cameras,
                self.gaussians,
                self.dataset,
                self.save_dir,
                self.pipeline_params,
                self.background,
                intrinsics=self.backend.intrinsics,
                height=self.backend.height,
                width=self.backend.width,
                kf_indices=kf_indices,
                iteration="before_opt",
                upsampling_method=config['Hierarchical']['upsampling_method'],
            )
            self.psnr_before = rendering_result_before.get("mean_psnr", None)

            # re-used the frontend queue to retrive the gaussians from the backend.
            while not frontend_queue.empty():
                frontend_queue.get()
            Log(f"[INFO] Total Gaussians before color refinement: {self.gaussians.get_anchor.shape[0]}")
            # 2) Run color refinement on backend and measure its time
            color_refine_start = time.time()
            backend_queue.put(["color_refinement"])
            while True:
                if frontend_queue.empty():
                    time.sleep(0.01)
                    continue
                data = frontend_queue.get()

                if data[0] == "color_refinement_done":
                    self.gaussians = data[1]
                    break
            color_refine_end = time.time()
            self.color_refine_time = color_refine_end - color_refine_start
            Log(f"[INFO] Color refinement time: {self.color_refine_time:.2f} s")
            # 3) Evaluate rendering & PSNR AFTER color refinement
            rendering_result_after = eval_rendering(
                self.frontend.cameras,
                self.gaussians,
                self.dataset,
                self.save_dir,
                self.pipeline_params,
                self.background,
                intrinsics=self.backend.intrinsics,
                height=self.backend.height,
                width=self.backend.width,
                kf_indices=kf_indices,
                iteration="after_opt",
                upsampling_method=config['Hierarchical']['upsampling_method'],
            )
            
            print("saving results in " + self.save_dir)
            
            # print(f"DEBUG: Attempting to save checkpoint to {self.save_dir}...", flush=True)
            if config["Results"].get("save_ckpt", False):
                self.gaussians.save_ckpt(self.save_dir)
            self.psnr_after = rendering_result_after.get("mean_psnr", None)
            Log(f"[INFO] Total Gaussians after color refinement: {self.gaussians.get_anchor.shape[0]}")
            if self.psnr_before is not None and self.psnr_after is not None:
                Log(
                    f"[Eval] PSNR before color refine: {self.psnr_before:.2f} dB, "
                    f"after color refine: {self.psnr_after:.2f} dB"
                )

        # === clear backend_queue before sending stop ===
        try:
            while not self.backend_queue.empty():
                self.backend_queue.get_nowait()
        except:
            pass

        # request backend to stop and wait (with timeout)
        try:
            self.backend_queue.put(["stop"])
        except Exception:
            pass

        # === clear frontend queue ===
        try:
            while not self.frontend_queue.empty():
                self.frontend_queue.get_nowait()
        except:
            pass
        
        # wait join with timeout, otherwise terminate
        self._join_process(self.backend_process, timeout=10.0, name="BackendProcess")

        Log("Backend stopped and joined the main thread")

        # stop GUI if running
        if self.use_gui and self.gui_process is not None:
            try:
                self.q_main2vis.put(gui_utils.GaussianPacket(finish=True))
            except Exception:
                pass
            self._join_process(self.gui_process, timeout=5.0, name="GUIProcess")
            Log("GUI Stopped and joined the main thread")

        # close queues
        for q in (self.backend_queue, self.frontend_queue, self.q_main2vis, self.q_vis2main):
            try:
                q.close()
                q.join_thread()
            except Exception:
                pass

        # final sync
        torch.cuda.synchronize()

    def _join_process(self, proc, timeout=5.0, name=None):
        """Try join with timeout and terminate if still alive."""
        if proc is None:
            return
        proc.join(timeout=timeout)
        if proc.is_alive():
            Log(f"[WARN] {name or proc.name} did not exit in {timeout}s; terminating.")
            try:
                proc.terminate()
            except Exception as e:
                Log(f"[ERROR] terminating {name}: {e}")
            proc.join(timeout=2.0)
            if proc.is_alive():
                Log(f"[ERROR] {name} still alive after terminate; killing process.")
                try:
                    os.kill(proc.pid, signal.SIGKILL)
                except Exception as e:
                    Log(f"[ERROR] kill failed: {e}")

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--config", type=str, required=True, help="Scene YAML configuration")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--kf_iter_num", type=int, default=None,
                        help="Override mapping iterations per keyframe; otherwise use YAML")

    parser.add_argument("--mode", choices=("full", "frontend"), default=None,
                        help="Override SLAM.mode; defaults to full without editing YAML")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Limit input frames for smoke tests; default is the entire sequence")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Explicit run directory (must be empty or not exist)")
    parser.add_argument("--feature-mode", "--feature_mode", choices=("single", "hybrid"), default=None)
    parser.add_argument("--loop-closure", action=BooleanOptionalAction, default=None)
    parser.add_argument("--hybrid-replacement", action=BooleanOptionalAction, default=None)
    args = parser.parse_args(sys.argv[1:])
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max_frames must be positive")

    mp.set_start_method("spawn", force=True)

    config = load_config(args.config)
    if args.feature_mode is not None:
        config.setdefault("tracking", {})["feature_mode"] = args.feature_mode
    if args.loop_closure is not None:
        config.setdefault("SLAM", {})["loop_closure"] = args.loop_closure
    if args.hybrid_replacement is not None:
        config.setdefault("tracking", {}).setdefault("hybrid", {})["apply_replacement"] = args.hybrid_replacement
    config["SLAM"]["use_heavy"] = False  # Loop closure always uses original light matching.
    config.setdefault("SLAM", {})["mode"] = args.mode or config["SLAM"].get("mode", "full")
    if config["SLAM"]["mode"] not in ("full", "frontend"):
        parser.error("SLAM.mode must be full or frontend")
    if args.max_frames is not None:
        config["SLAM"]["max_frames"] = args.max_frames
    if args.output_dir or config["SLAM"]["mode"] == "frontend":
        config["Results"]["save_results"] = True
    save_dir = None

    if args.kf_iter_num is not None:
        Log(f"Overriding kf_iter_num to {args.kf_iter_num}")
        config['Training']['kf_iter_num'] = args.kf_iter_num
    
    if args.eval:
        Log("Running MonoGS in Evaluation Mode")
        Log("Following config will be overriden")
        Log("\tsave_results=True")
        config["Results"]["save_results"] = True
        Log("\tuse_gui=False")
        config["Results"]["use_gui"] = False
        Log("\teval_rendering=True")
        config["Results"]["eval_rendering"] = True
        Log("\tuse_wandb=True")
        config["Results"]["use_wandb"] = True

    if config["SLAM"]["mode"] == "frontend":
        config["Results"]["use_gui"] = False
        config["Results"]["eval_rendering"] = False
    Log(f"Execution mode: {config['SLAM']['mode']}")

    if config["Results"]["save_results"]:
        mkdir_p(config["Results"]["save_dir"])
        current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        if config['SLAM']['loop_closure']:
            current_datetime += '-LC'
        else:
            current_datetime += '-No-LC'
            
        # Use the frame-index filename to identify KITTI-360 sequences.
        index_path = config["Dataset"].get("index_path", None)
        if index_path is not None:
            
            
            experiment_name = os.path.splitext(os.path.basename(index_path))[0]
        else:
            
            
            raw_path = config["Dataset"]["color_path"].rstrip("/")
            path_parts = raw_path.split("/")
            
            experiment_name = path_parts[-3] + "_" + path_parts[-2] + "_" + path_parts[-1]

        save_dir = os.path.join(
            config["Results"]["save_dir"], experiment_name, current_datetime
        )
        if config["SLAM"]["mode"] == "frontend":
            save_dir += "-frontend"
        if args.output_dir:
            save_dir = os.path.abspath(args.output_dir)
            if os.path.exists(save_dir) and os.listdir(save_dir):
                parser.error("--output_dir must be empty or not exist")
        config["Results"]["save_dir"] = save_dir
        mkdir_p(save_dir)
        with open(os.path.join(save_dir, "config.yml"), "w") as file:
            yaml.dump(config, file)
        Log("saving results in " + save_dir)


    # start timing & reset peak mem
    torch.cuda.reset_peak_memory_stats()
    time_start = time.time()

    slam = None
    exit_code = 0
    try:
        slam = SLAM(config, save_dir=save_dir)
    except KeyboardInterrupt:
        exit_code = 130
        Log("[INFO] KeyboardInterrupt received. Attempting graceful shutdown...")
    except Exception as e:
        exit_code = 1
        traceback.print_exc()
        Log(f"[ERROR] Exception in main: {e}")
    finally:
        # Ensure processes cleaned up even if SLAM.__init__ crashed partway
        Log("[INFO] Main cleanup: terminating leftover processes & queues if any.")
        # terminate any remaining child processes
        for p in mp.active_children():
            try:
                Log(f"[INFO] Terminating active child: pid={p.pid}, name={p.name}")
                p.terminate()
                p.join(timeout=2.0)
            except Exception:
                pass

        # Attempt to close any queues possibly created by SLAM
        try:
            if slam is not None:
                for q in (getattr(slam, 'backend_queue', None),
                          getattr(slam, 'frontend_queue', None),
                          getattr(slam, 'q_main2vis', None),
                          getattr(slam, 'q_vis2main', None)):
                    if q is not None:
                        try:
                            q.close()
                            q.join_thread()
                        except Exception:
                            pass
        except Exception:
            pass

        # Clear CUDA and GC
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()

    # sync and measure
    try:
        torch.cuda.synchronize()
    except Exception:
        pass

    time_end = time.time()
    time_total = time_end - time_start
    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0  # MB

    # Collect phase timings and rendering metrics.
    if slam is not None:
        slam_time = getattr(slam, "slam_time", time_total)
        color_refine_time = getattr(slam, "color_refine_time", 0.0)
        psnr_before = getattr(slam, "psnr_before", None)
        psnr_after = getattr(slam, "psnr_after", None)
        eval_rendering_flag = getattr(slam, "eval_rendering", False)
    else:
        slam_time = time_total
        color_refine_time = 0.0
        psnr_before = None
        psnr_after = None
        eval_rendering_flag = False

    # Persist timing and memory measurements.
    if save_dir is not None:
        time_mem_txt = os.path.join(save_dir, "time_memory.txt")
        try:
            with open(time_mem_txt, "w") as f:
                f.write(f"Total time (script): {time_total:.2f} s\n")
                f.write(f"SLAM time (before color refine): {slam_time:.2f} s\n")
                if eval_rendering_flag:
                    f.write(f"Color refinement time: {color_refine_time:.2f} s\n")
                f.write(f"Peak GPU memory: {peak_memory:.2f} MiB\n")
                if psnr_before is not None:
                    f.write(f"PSNR before color refine: {psnr_before:.2f} dB\n")
                if psnr_after is not None:
                    f.write(f"PSNR after color refine: {psnr_after:.2f} dB\n")
            Log(f"[INFO] Saved time and memory info to {time_mem_txt}")
        except Exception:
            Log("[WARN] Failed to write time_memory.txt")
    sys.exit(exit_code)
