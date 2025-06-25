import numpy as np
import torch
import shutil
import os
import cv2
import json
import figs.utilities.trajectory_helper as th

from figs.control.base_controller import BaseController
from figs.render.gsplat import GSplat
from figs.dynamics.model_equations import export_quadcopter_ode_model
from figs.dynamics.model_specifications import generate_specifications
from acados_template import AcadosSimSolver, AcadosSim
from typing import Dict,List,Type,Union

# from SaferSplat.cbf.cbf_utils import CBF
# from scipy.spatial.transform import Rotation

class Flight():
    def __init__(self, rollout_config:Dict[str,Union[int,float,List[float]]],
                 frame_config:Dict[str,Union[int,float,List[float]]], name:str='flyer') -> None:
        """
        Flying class for simulating drone flights.

        Args:
            - rollout_config:   Configuration dictionary for the rollout.
            - frame_config:     Configuration dictionary for the frame.
            - name:             Name of the flight.

        Variables:
            - nx: Number of states in the system.
            - nu: Number of controls in the system.
            - hz_sim: Simulation rate.
            - t_dly: Delay in control.
            - mu_md: Mean of the model noise.
            - std_md: Standard deviation of the model noise.
            - mu_sn: Mean of the sensor noise.
            - std_sn: Standard deviation of the sensor noise.
            - use_fusion: Use sensor model fusion.
            - Wf: Fusion weights.
            - drn_spec: Drone specifications.
            - simulator: Acados simulator.
            - code_export_path: Path to the generated code.
            - simulator_path: Path to the simulator json file.
        
        """

        # Some useful intermediate variables
        drn_spec = generate_specifications(frame_config)
        sim_json = 'acados_sim_nlp_'+name+'.json'

        sim = AcadosSim()
        sim.model = export_quadcopter_ode_model(drn_spec["m"],drn_spec["tn"])  
        sim.solver_options.T = 1/rollout_config["hz_sim"]
        sim.solver_options.integrator_type = 'IRK'
        sim.code_export_directory = os.path.join(sim.code_export_directory,name)

        # Class variables
        self.nx,self.nu = sim.model.x.size()[0],sim.model.u.size()[0]
        self.hz_sim = rollout_config["hz_sim"]
        self.t_dly = rollout_config["delay"]
        self.mu_md = np.array(rollout_config["model_noise"]["mean"])
        self.std_md = np.array(rollout_config["model_noise"]["std"])
        self.mu_sn = np.array(rollout_config["sensor_noise"]["mean"])
        self.std_sn = np.array(rollout_config["sensor_noise"]["std"])
        self.use_fusion = False#rollout_config["sensor_model_fusion"]["use_fusion"]
        self.Wf = np.diag(rollout_config["sensor_model_fusion"]["weights"])
        self.drn_spec = drn_spec
        self.simulator = AcadosSimSolver(sim, json_file=sim_json, verbose=False)
        
        self.code_export_path = sim.code_export_directory
        self.simulator_path = os.path.join(os.getcwd(),sim_json)

        # Clear the generated code
        self.clear_generated_code()

    def simulate(self,controller:Type[BaseController],gsplat:GSplat,
                 t0:float,tf:int,x0:np.ndarray,
                 obj:Union[None,np.ndarray]=None):
        """
        Method to simulate the drone flight.

        Args:
            - controller: Controller for the drone.
            - camera: Camera object for rendering images.
            - gsplat: GSplat object for rendering images.
            - t0: Initial time.
            - tf: Final time.
            - x0: Initial state.
            - obj: Object to track.

        Returns:
            - Tro: Time vector.
            - Xro: State vector.
            - Uro: Control vector.
            - Iro: Image vector.
            - Tsol: Solution time vector.
            - Adv: Advisor vector (if used).

        """
        
        # Simulation Variables
        dt = np.round(tf-t0)
        Nsim = int(dt*self.hz_sim)
        Nctl = int(dt*controller.hz)
        n_sim2ctl = int(self.hz_sim/controller.hz)
        n_delay = int(self.t_dly*self.hz_sim)
        cam_cfg = self.drn_spec["camera"]
        height,width,channels = cam_cfg["height"],cam_cfg["width"],cam_cfg["channels"]
        T_c2b = self.drn_spec["T_c2b"]

        # Extract sensor and model parameters
        mu_md  = self.mu_md*(1/n_sim2ctl)         # Scale model mean noise to control rate
        std_md = self.std_md*(1/n_sim2ctl)        # Scale model std noise to control rate
        mu_sn = 1.0*self.mu_sn
        std_sn = 1.0*self.std_sn
        Wf_sn,Wf_md = self.Wf,1-self.Wf

        # Rollout Variables
        Tro,Xro,Uro = np.zeros(Nctl+1),np.zeros((self.nx,Nctl+1)),np.zeros((self.nu,Nctl))
        Iro = np.zeros((Nctl,height,width,channels),dtype=np.uint8)
        Xro[:,0] = x0

        # Diagnostics Variables
        Tsol = np.zeros((4,Nctl))
        Adv = np.zeros((self.nu,Nctl))
        
        # Transient Variables
        xcr,xpr,xsn = x0.copy(),x0.copy(),x0.copy()
        ucm = np.array([-self.drn_spec['m']/self.drn_spec['tn'],0.0,0.0,0.0])
        udl = np.hstack((ucm.reshape(-1,1),ucm.reshape(-1,1)))
        zcr = torch.zeros(controller.nzcr) if isinstance(controller.nzcr, int) else None

        # Instantiate camera object
        camera = gsplat.generate_output_camera(cam_cfg)

        # Rollout
        for i in range(Nsim):
            # Get current time and state
            tcr = t0+i/self.hz_sim

            # Control
            if i % n_sim2ctl == 0:
                # Get current image
                Tb2w = th.xv_to_T(xcr)
                T_c2w = Tb2w@T_c2b
                icr = gsplat.render_rgb(camera,T_c2w)

                # Add sensor noise and syncronize estimated state
                if self.use_fusion:
                    xsn += np.random.normal(loc=mu_sn,scale=std_sn)
                    xsn = Wf_sn@xsn + Wf_md@xcr
                else:
                    xsn = xcr + np.random.normal(loc=mu_sn,scale=std_sn)
                xsn[6:10] = th.obedient_quaternion(xsn[6:10],xpr[6:10])

                # Generate controller command
                ucm,zcr,adv,tsol = controller.control(tcr,xsn,ucm,obj,icr,zcr)
                # Update delay buffer
                udl[:,0] = udl[:,1]
                udl[:,1] = ucm

            # Extract delayed command
            uin = udl[:,0] if i%n_sim2ctl < n_delay else udl[:,1]
            # print(f"Control at step {i}: {uin}")

            # Simulate both estimated and actual states
            xcr = self.simulator.simulate(x=xcr,u=uin)
            if self.use_fusion:
                xsn = self.simulator.simulate(x=xsn,u=uin)

            # Add model noise
            xcr = xcr + np.random.normal(loc=mu_md,scale=std_md)
            xcr[6:10] = th.obedient_quaternion(xcr[6:10],xpr[6:10])

            # Update previous state
            xpr = xcr
            
            # Store values
            if i % n_sim2ctl == 0:
                k = i//n_sim2ctl

                Iro[k,:,:,:] = icr
                Tro[k] = tcr
                Xro[:,k+1] = xcr
                Uro[:,k] = ucm
                Tsol[:,k] = tsol
                Adv[:,k] = adv

        # Log final time
        Tro[Nctl] = t0+Nsim/self.hz_sim

        return Tro,Xro,Uro,Iro,Tsol,Adv
    
    

    def control_to_accl(self, F, x_cr):
        """
        Convert control inputs to world-frame acceleration.
        
        Args:
            F: scalar thrust in body frame (acts along body Z+)
            q_current: current orientation quaternion (x, y, z, w)
            mass: quadcopter mass
            gravity: gravitational acceleration (default 9.81)
        
        Returns:
            acc_world: np.array([ax, ay, az])
        """
        g = 9.81

        R = th.xv_to_T(x_cr)[:3,:3]

        T_b = np.array([0.0, 0.0, self.drn_spec['tn']*F/self.drn_spec['m']])

        # Rotate to world frame
        T_w = R @ T_b

        # Net acceleration
        accl = T_w + np.array([0.0, 0.0, g])

        return accl
    
    def accl_control(self, accl, x_cr):
        g = np.array([0, 0, 9.81])
        accl[2] = 1.1*accl[2]
        total_acc = accl - g
        z_body_des = -total_acc / np.linalg.norm(total_acc)

        x_body = np.array([0.0, np.sin(1.57), 0.0])
        y_body_des = np.cross(z_body_des, x_body)
        y_body_des /= np.linalg.norm(y_body_des)
        x_body_des = np.cross(y_body_des, z_body_des)
        R_des = np.column_stack((x_body_des, y_body_des, z_body_des))
 
        R_cr = th.xv_to_T(x_cr)[:3,:3]
        # print(R_cr)
        # print(R_des)
        # ax, ay, az = accl
        # wx = (ax*np.sin(-1.57) - ay*np.cos(-1.57))/total_acc
        # wy= (ax*np.cos(-1.57) + ay*np.sin(-1.57))/total_acc

        # R_des = 

        # Compute orientation error
        R_err = R_des @ R_cr.T
        skew_err = 0.5 * (R_err - R_err.T)
        ang_err = np.array([skew_err[0, 2], -skew_err[2, 1], skew_err[1, 0]])

        # Thrust
        F = -np.linalg.norm(self.drn_spec['m'] * total_acc / self.drn_spec['tn'])

        omega = 0.6*ang_err

        return np.concatenate(([F], omega))

    
    # def simulate_cbf(self,controller:Type[BaseController],gsplat:GSplat,cbf:CBF,
    #              t0:float,tf:int,x0:np.ndarray,
    #              obj:Union[None,np.ndarray]=None):
    #     """
    #     Method to simulate the drone flight.

    #     Args:
    #         - controller: Controller for the drone.
    #         - camera: Camera object for rendering images.
    #         - gsplat: GSplat object for rendering images.
    #         - t0: Initial time.
    #         - tf: Final time.
    #         - x0: Initial state.
    #         - obj: Object to track.

    #     Returns:
    #         - Tro: Time vector.
    #         - Xro: State vector.
    #         - Uro: Control vector.
    #         - Iro: Image vector.
    #         - Tsol: Solution time vector.
    #         - Adv: Advisor vector (if used).

    #     """
        
    #     # Simulation Variables
    #     dt = np.round(tf-t0)
    #     Nsim = int(dt*self.hz_sim)
    #     Nctl = int(dt*controller.hz)
    #     n_sim2ctl = int(self.hz_sim/controller.hz)
    #     n_delay = int(self.t_dly*self.hz_sim)
    #     cam_cfg = self.drn_spec["camera"]
    #     height,width,channels = cam_cfg["height"],cam_cfg["width"],cam_cfg["channels"]
    #     T_c2b = self.drn_spec["T_c2b"]


    #     # Rollout Variables
    #     Tro,Xro,Uro = np.zeros(Nctl+1),np.zeros((self.nx,Nctl+1)),np.zeros((self.nu,Nctl))
    #     Hro, Ucbfro = np.zeros(Nctl), np.zeros(Nctl) 
    #     Iro = np.zeros((Nctl,height,width,channels),dtype=np.uint8)
    #     x0[3:] = np.array([0,0,0,0,0,0.7068,0.70738])
    #     Xro[:,0] = x0
    #     xf = np.array([-0.5,0.5,-0.6])

    #     # Diagnostics Variables
    #     Tsol = np.zeros((4,Nctl))
    #     Adv = np.zeros((self.nu,Nctl))
        
    #     # Transient Variables
    #     xcr,xpr = x0.copy(),x0.copy()
    #     ucm = np.array([-self.drn_spec['m']/self.drn_spec['tn'],0.0,0.0,0.0])
    #     udl = np.hstack((ucm.reshape(-1,1),ucm.reshape(-1,1)))
    #     zcr = torch.zeros(controller.nzcr) if isinstance(controller.nzcr, int) else None

    #     # Instantiate camera object
    #     camera = gsplat.generate_output_camera(cam_cfg)


    #     # Rollout
    #     for i in range(Nsim):
    #         # Get current time and state
    #         tcr = t0+i/self.hz_sim

    #         # Control
    #         if i % n_sim2ctl == 0:
    #             # Get current image
    #             Tb2w = th.xv_to_T(xcr)
    #             T_c2w = Tb2w@T_c2b
    #             icr = gsplat.render_rgb(camera,T_c2w)


    #             # Generate controller command
    #             # ucm,zcr,adv,tsol = controller.control(tcr,xsn,ucm,obj,icr,zcr)
    #             # print("ucm:", ucm)
    #             # Update delay buffer
    #             ucbf = cbf.solve_QP(torch.tensor(xcr[:6],dtype=torch.float32, device='cuda'),torch.tensor(0.2*(xf-xcr[:3]),dtype=torch.float32, device='cuda')).cpu().numpy()
    #             # print("ucbf1",ucbf)
    #             # print("des:", 0.5*(xf-xcr[:3]))
    #             # hro,_, _, _ = cbf.gsplat.query_distance(torch.tensor(xcr[:3],dtype=torch.float32, device='cuda'), radius=0.4)
    #             # hro = np.min(hro.cpu().numpy())
    #             # ucbfro = np.linalg.norm(ucbf-0.2*(xf-xcr[:3]))
    #             ucbf = self.accl_control(ucbf, xcr)
    #             udl[:,0] = udl[:,1]
    #             udl[:,1] = ucbf
    #             # print("ucbf", ucbf)

    #         # Extract delayed command
    #         uin = udl[:,0] if i%n_sim2ctl < n_delay else udl[:,1]
    #         # print(f"Control at step {i}: {uin}")

    #         # Simulate both estimated and actual states
    #         xcr = self.simulator.simulate(x=xcr,u=uin)
        
    #         # Add model noise
    #         xcr = xcr
    #         xcr[6:10] = th.obedient_quaternion(xcr[6:10],xpr[6:10])

    #         # Update previous state
    #         xpr = xcr
            
    #         # Store values
    #         if i % n_sim2ctl == 0:
    #             k = i//n_sim2ctl

    #             Iro[k,:,:,:] = icr
    #             Tro[k] = tcr
    #             Xro[:,k+1] = xcr
    #             Uro[:,k] = ucm
    #             # Ucbfro[k] = ucbfro
    #             # Hro[k] = hro
    #             # Tsol[:,k] = tsol
    #             # Adv[:,k] = adv

    #     # Log final time
    #     Tro[Nctl] = t0+Nsim/self.hz_sim

    #     return Tro,Xro,Uro,Iro,Tsol,Adv, Ucbfro, Hro
    
    def generate_framepair2posedelta_samples(self,gsplat,scene: str,
        lower_bound,upper_bound,num_samples=5000,
        save_path="/home/purush/CroCo/data/pair_pose"
    ):
        """
        Generates image pairs and relative pose data using GSplat.
        Each sample contains:
        - Two rendered images from nearby poses.
        - A 6-DoF relative pose (translation + rotation delta).
        Splits into train (70%), validation (20%), and test (10%).

        Args:
            gsplat: GSplat renderer instance.
            scene: Scene identifier for saving.
            lower_bound: 3D minimum sample coordinates (x, y, z).
            upper_bound: 3D maximum sample coordinates (x, y, z).
            num_samples: Total samples to generate.
            save_path: Root directory to save images and annotations.
        """

        # Create output directories
        scene_path = os.path.join(save_path, scene)
        os.makedirs(scene_path, exist_ok=True)

        split_paths = {
            "train": os.path.join(scene_path, "train"),
            "validation": os.path.join(scene_path, "validation"),
            "test": os.path.join(scene_path, "test")
        }
        for path in split_paths.values():
            os.makedirs(path, exist_ok=True)

        # Camera setup
        cam_cfg = self.drn_spec["camera"]
        T_c2b = self.drn_spec["T_c2b"]
        camera = gsplat.generate_output_camera(cam_cfg)

        samples = {"train": [], "validation": [], "test": []}

        for i in range(num_samples):
            xcr1 = np.zeros(10)
            xcr2 = np.zeros(10)

            pos1 = np.random.uniform(lower_bound, upper_bound)
            delta_pos = np.random.uniform(-0.2, 0.2, size=3)
            pos2 = pos1 + delta_pos

            xcr1[0:3] = pos1
            xcr2[0:3] = pos2

            roll1, pitch1, yaw1 = np.random.uniform(-np.pi/18, np.pi/18, size=2).tolist() + [np.random.uniform(-np.pi, np.pi)]
            delta_roll, delta_pitch, delta_yaw = np.random.uniform(-np.pi/36, np.pi/36, size=2).tolist() + [np.random.uniform(-np.pi/18, np.pi/18)] 
            roll2, pitch2, yaw2 = roll1 + delta_roll, pitch1 + delta_pitch, yaw1 + delta_yaw

            xcr1[6:10] = th.rpy_to_quat(roll1, pitch1, yaw1)
            xcr2[6:10] = th.rpy_to_quat(roll2, pitch2, yaw2)

            T_c2w_1 = th.xv_to_T(xcr1) @ T_c2b
            T_c2w_2 = th.xv_to_T(xcr2) @ T_c2b

            frame1 = gsplat.render_rgb(camera, T_c2w_1)
            frame2 = gsplat.render_rgb(camera, T_c2w_2)

            if i < 0.9 * num_samples:
                split = "train"
            elif i < 0.95 * num_samples:
                split = "validation"
            else:
                split = "test"

            subset_path = split_paths[split]

            frame1_path = os.path.join(subset_path, f"frame{i}_1.png")
            frame2_path = os.path.join(subset_path, f"frame{i}_2.png")
            cv2.imwrite(frame1_path, frame2)
            cv2.imwrite(frame2_path, frame1)

            relative_pose = np.zeros(6)
            relative_pose[0:3] = delta_pos
            relative_pose[3:] = [delta_roll, delta_pitch, delta_yaw]

            samples[split].append((frame1_path, frame2_path, relative_pose.tolist()))

        for split, data in samples.items():
            output_file = os.path.join(scene_path, f"{split}_pairs.json")
            with open(output_file, "w") as f:
                json.dump(data, f, indent=4)
    

    def generate_frame2pose_samples(self, gsplat: GSplat, scene: str,
                                lower_bound, upper_bound, num_samples=100000, 
                                save_path="/home/purush/CroCo/data/frame_pose_euler"):
        """
        Generates 'num_samples' frames and associated poses from GSplat.
        Splits samples into train (70%), validation (20%), and test (10%).

        Args:
            lower_bound: Minimum scene coordinate for sampling.
            upper_bound: Maximum scene coordinate for sampling.
            gsplat: GSplat renderer instance.
            num_samples: Number of frames to generate.
            save_path: Directory to save frames and poses.

        Returns:
            None
        """
        save_path = os.path.join(save_path, scene)
        train_path = os.path.join(save_path, "train")
        validation_path = os.path.join(save_path, "validation")
        test_path = os.path.join(save_path, "test")

        os.makedirs(train_path, exist_ok=True)
        os.makedirs(validation_path, exist_ok=True)
        os.makedirs(test_path, exist_ok=True)

        cam_cfg = self.drn_spec["camera"]
        T_c2b = self.drn_spec["T_c2b"]

        # Instantiate camera object
        camera = gsplat.generate_output_camera(cam_cfg)

        samples = {"train": [], "validation": [], "test": []}

        for i in range(num_samples):
            xcr = np.zeros(10)
            xcr[0:3] = np.random.uniform(lower_bound, upper_bound)
            roll = np.random.uniform(-np.pi/18, np.pi/18)
            pitch = np.random.uniform(-np.pi/18, np.pi/18)
            yaw = np.random.uniform(-np.pi, np.pi)
            
            quaternion = th.rpy_to_quat(roll, pitch, yaw)
            xcr[6:10] = quaternion

            Tb2w = th.xv_to_T(xcr)
            T_c2w = Tb2w @ T_c2b
            frame = gsplat.render_rgb(camera, T_c2w)

            if i < 0.9 * num_samples:
                subset = "train"
                subset_path = train_path
            elif i < 0.95 * num_samples:
                subset = "validation"
                subset_path = validation_path
            else:
                subset = "test"
                subset_path = test_path

            frame_path = os.path.join(subset_path, f"frame_{i}.png")
            cv2.imwrite(frame_path, frame)

            pose = np.zeros(6)
            pose[0:3] = xcr[0:3]
            pose[3:] = np.array([roll, pitch, yaw])
            samples[subset].append((frame_path, pose.tolist()))

        for subset in ["train", "validation", "test"]:
            with open(os.path.join(save_path, f"{subset}_poses.json"), "w") as f:
                json.dump(samples[subset], f, indent=4)

    def generate_pose_frame(self, gsplat: GSplat, pose:np.ndarray,
                            save_path="/home/purush/FiGS-Examples/notebooks"):
        
        cam_cfg = self.drn_spec["camera"]

        T_c2b = self.drn_spec["T_c2b"]
        # Instantiate camera object
        camera = gsplat.generate_output_camera(cam_cfg)

        xcr = np.zeros(10)
        xcr[0:3] = pose[0:3]
        xcr[6:10] = pose[3:7]

        Tb2w = th.xv_to_T(xcr)
        T_c2w = Tb2w @ T_c2b
        frame = gsplat.render_rgb(camera, T_c2w)

        frame_path = os.path.join(save_path, f"rendered_image.png")
        cv2.imwrite(frame_path, frame)

        return frame_path


    def clear_generated_code(self):
        """
        Method to clear the generated code and files to ensure the code is recompiled correctly each time.
        """

        # Clear the generated code
        try:
            os.remove(self.simulator_path)
            shutil.rmtree(self.code_export_path)
        except:
            pass
        
        # Clear the parent directory if empty
        parent_dir_path = os.path.dirname(self.code_export_path)
        if not os.listdir(parent_dir_path) and (os.path.basename(parent_dir_path) == 'c_generated_code'):
            shutil.rmtree(parent_dir_path)