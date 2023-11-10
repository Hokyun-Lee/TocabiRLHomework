import numpy as np
from gym.envs.mujoco import mujoco_env
from gym import utils
from math import exp, sin, cos, pi
import time
from pyquaternion import Quaternion
from . import op3_walk_env
from .utils.cubic import cubic
from .utils.lpf import lpf
from .utils.rotation import quat2fixedXYZ

GroundCollisionCheckBodyList = ["body_link",\
            "l_hip_yaw_link", "l_hip_roll_link", "l_hip_pitch_link", "l_knee_link",\
            "r_hip_yaw_link", "r_hip_roll_link", "r_hip_pitch_link", "r_knee_link",\
            "l_sho_pitch_link", "l_sho_roll_link", "l_el_link", \
            "r_sho_pitch_link", "r_sho_roll_link", "r_el_link"]

SelfCollisionCheckBodyList = GroundCollisionCheckBodyList + ["l_ank_pitch_link", "l_ank_roll_link", "r_ank_pitch_link", "r_ank_roll_link"]

ObstacleList = ["obstacle1", "obstacle2", "obstacle3", "obstacle4", "obstacle5", "obstacle6", "obstacle7", "obstacle8", "obstacle9"]

Kp = np.array([2000.0, 5000.0, 4000.0, 3700.0, 3200.0, 3200.0,
     2000.0, 5000.0, 4000.0, 3700.0, 3200.0, 3200.0,
     400.0, 100.0, 100.0,
     100.0, 100.0,
     400.0, 100.0, 100.0])

Kv = np.array([15.0, 50.0, 20.0, 25.0, 24.0, 24.0,
     15.0, 50.0, 20.0, 25.0, 24.0, 24.0,
     10.0, 3.0, 3.0,
     2.0, 2.0,
     10.0, 3.0, 3.0])

control_freq_scale = 1

class Op3Env(op3_walk_env.Op3Env):
    def __init__(self, frameskip=int(8/control_freq_scale)):
        super(Op3Env, self).__init__('op3.xml', frameskip)
        # utils.EzPickle.__init__(self)
        for id in GroundCollisionCheckBodyList:
            self.ground_collision_check_id.append(self.model.body_name2id(id))
        for id in SelfCollisionCheckBodyList:
            self.self_collision_check_id.append(self.model.body_name2id(id))
        self.ground_id.append(0)
        self.right_foot_id.append(self.model.body_name2id("r_ank_roll_link"))
        self.left_foot_id.append(self.model.body_name2id("l_ank_roll_link"))
        # for id in ObstacleList:
        #     self.ground_id.append(self.model.body_name2id(id))
        print("Collision Check ID", self.ground_collision_check_id)
        print("Self Collision Check ID", self.self_collision_check_id)
        print("Ground ID", self.ground_id)
        print("R Foot ID",self.model.body_name2id("r_ank_roll_link"))
        print("L Foot ID",self.model.body_name2id("l_ank_roll_link"))

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel

        orientation = Quaternion(array=qpos[3:7])
        
        # orientation_noise = np.random.uniform(-0.034, 0.034,3)
        # orientation = orientation * Quaternion(axis=(1.0, 0.0, 0.0), radians=orientation_noise[0]) * \
        #                 Quaternion(axis=(0.0, 1.0, 0.0), radians=orientation_noise[1]) * Quaternion(axis=(0.0, 0.0, 1.0), radians=orientation_noise[2])

        fixed_angle = quat2fixedXYZ(orientation.elements)
        fixed_angle[:] = fixed_angle[:] + self.quat_bias

        mocap_cycle_period = self.mocap_data_num* self.mocap_cycle_dt
        phase = np.array((self.init_mocap_data_idx + self.time % mocap_cycle_period / self.mocap_cycle_dt) % self.mocap_data_num / self.mocap_data_num)
        sin_phase = np.array(sin(2*pi*phase))
        cos_phase = np.array(cos(2*pi*phase))     

        # cur_obs = np.concatenate([[euler_angle[0], euler_angle[1], euler_angle[2]],
        #     (self.qpos_noise).flatten(),
        #     self.qvel_lpf.flatten(),
        #     qvel[3:6].flatten(),
        #     sin_phase.flatten(),
        #     cos_phase.flatten(),
        #     [self.target_vel]])

        cur_obs = np.concatenate([[fixed_angle[0], fixed_angle[1], fixed_angle[2]],
                    (self.qpos_noise[0:12] + self.q_bias).flatten(),
                    (self.qvel_noise[0:12]).flatten(),
                    # self.qvel_lpf[0:12].flatten(),
                    # qvel[3:6].flatten(),
                    sin_phase.flatten(),
                    cos_phase.flatten(),
                    [self.target_vel[0]],[self.target_vel[1]]])

        self.action_last = np.copy(self.action_cur)     
        self.qvel_pre = np.copy(qvel[6:])

        cur_obs = (cur_obs - self.obs_mean) / np.sqrt(self.obs_var + 1e-8)
        
        if (self.epi_len == 0 or self.obs_buf == []):
            for _ in range(self.num_obs_hist):
                for _ in range(self.num_obs_skip):
                    self.obs_buf.append(cur_obs)
                    self.action_buf.append(np.array(self.action_raw, dtype=np.float64))
        
        self.obs_buf[0:self.num_obs_skip*self.num_obs_hist-1] = self.obs_buf[1:self.num_obs_skip*self.num_obs_hist]
        self.obs_buf[-1] = cur_obs
        self.action_buf[0:self.num_obs_skip*self.num_obs_hist-1] = self.action_buf[1:self.num_obs_skip*self.num_obs_hist]
        self.action_buf[-1] = np.array(self.action_raw, dtype=np.float64)

        obs = []
        for i in range(self.num_obs_hist):
            obs.append(self.obs_buf[self.num_obs_skip*(i+1)-1])
        
        act = []
        for i in range(self.num_obs_hist-1):
            act.append(self.action_buf[self.num_obs_skip*(i+1)])

        return np.concatenate([np.array(obs).flatten(), np.array(act).flatten()])


    def step(self, a):
        self.action_raw = np.copy(a)
        a = a * self.action_high
        done_by_early_stop = False
        self.action_cur = a[0:-1] * self.motor_constant_scale
        # print("Action: ", a)
        # a[:] = 0.0

        mocap_cycle_period = self.mocap_data_num* self.mocap_cycle_dt

        local_time = self.time % mocap_cycle_period
        local_time_plus_init = (local_time + self.init_mocap_data_idx*self.mocap_cycle_dt) % mocap_cycle_period
        self.mocap_data_idx = (self.init_mocap_data_idx + int(local_time / self.mocap_cycle_dt)) % self.mocap_data_num
        next_idx = self.mocap_data_idx + 1 
        
        target_data_qpos = np.zeros_like(a)    
        target_data_qpos = cubic(local_time_plus_init, self.mocap_data[self.mocap_data_idx,0], self.mocap_data[next_idx,0], self.mocap_data[self.mocap_data_idx,1:34], self.mocap_data[next_idx,1:34], 0.0, 0.0)

        if (self.perturbation_on):
            if (self.epi_len % (control_freq_scale*2000) == self.perturb_timing):
                self.magnitude = np.random.uniform(0, 250)
                theta = np.random.uniform(0, 2*pi)
                self.new_xfrc[1,0] = self.magnitude * cos(theta)
                self.new_xfrc[1,1] = self.magnitude * sin(theta)
                self.pert_duration = control_freq_scale*np.random.randint(1, 50)
                self.cur_pert_duration = 0
            
            if (self.cur_pert_duration < self.pert_duration):
                self.sim.data.xfrc_applied[:] = self.new_xfrc
                self.cur_pert_duration = self.cur_pert_duration + 1
            else:
                self.sim.data.xfrc_applied[:] = np.zeros_like(self.sim.data.xfrc_applied)

        if (self.spec is not None):
            if (self.epi_len % int(self.spec.max_episode_steps/4) == int(self.spec.max_episode_steps/4)-1):
                self.vel_change_duration = np.random.randint(1, 250)
                self.cur_vel_change_duration = 0  
                self.start_target_vel = np.copy(self.target_vel)
                self.final_target_vel = np.array([np.random.uniform(-0.2, 0.5), np.random.uniform(-0.2, 0.2)])         
            if (self.cur_vel_change_duration < self.vel_change_duration):
                self.target_vel = self.start_target_vel + (self.final_target_vel-self.start_target_vel) * self.cur_vel_change_duration / self.vel_change_duration
                self.cur_vel_change_duration = self.cur_vel_change_duration + 1
            else:
                self.target_vel = np.copy(self.target_vel)

        # Simulation
        for _ in range(self.frame_skip):
            upper_torque = Kp[12:]*(self.init_q_desired[19:] - self.qpos_noise[12:]) + Kv[12:]*(-self.qvel_noise[12:])
            self.action_log.append(np.concatenate([self.action_cur, upper_torque]))
            if (len(self.action_log) < self.action_delay):
                a_idx = -len(self.action_log)
            else:
                a_idx = -self.action_delay
            self.do_simulation(self.action_log[a_idx],1) 
            qpos = self.sim.data.qpos[7:]
            # self.qpos_noise = qpos + np.random.uniform(-0.00001, 0.00001, len(qpos))
            self.qpos_noise = qpos + np.clip(np.random.normal(0, 0.00004 / 3.0, len(qpos)), -0.00004, 0.00004)
            self.qvel_noise = (self.qpos_noise - self.qpos_pre) / self.model.opt.timestep
            self.qpos_pre = np.copy(self.qpos_noise)
            # self.qvel_lpf = lpf(self.qvel_noise, self.qvel_lpf, 1/self.model.opt.timestep, 4.0)

        # init_q_pos = np.copy(self.init_q_desired)
        # init_q_pos[7:] = np.concatenate([target_data_qpos[0:12], self.init_q_desired[19:27]])
        # init_qvel = np.copy(self.init_qvel)
        # # init_qvel[0] = np.random.uniform(0.0, 0.3)
        # self.set_state(init_q_pos, init_qvel)  
        self.time += self.dt
        self.time += a[-1]
 
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        
        # Collision Check
        geom1 = np.zeros(self.sim.data.ncon)
        geom2 = np.zeros(self.sim.data.ncon)
        for i in range(self.sim.data.ncon):
            geom1[i] = self.model.geom_bodyid[self.sim.data.contact[i].geom1]
            geom2[i] = self.model.geom_bodyid[self.sim.data.contact[i].geom2]
            
        if (np.in1d(geom1, self.ground_id) * \
                np.in1d(geom2, self.ground_collision_check_id)).any() or \
            (np.in1d(geom2, self.ground_id) * \
                np.in1d(geom1, self.ground_collision_check_id)).any():
            done_by_early_stop = True # Ground-Body contact
        if (np.in1d(geom1, self.self_collision_check_id) * \
                np.in1d(geom2, self.self_collision_check_id)).any():
            done_by_early_stop = True # Self Collision contact

        left_foot_contact = False
        if (np.in1d(geom1, self.ground_id) * \
                np.in1d(geom2, self.left_foot_id)).any() or \
            (np.in1d(geom2, self.left_foot_id) * \
                np.in1d(geom1, self.ground_id)).any():
            left_foot_contact = True 
        right_foot_contact = False
        if (np.in1d(geom1, self.ground_id) * \
                np.in1d(geom2, self.right_foot_id)).any() or \
            (np.in1d(geom2, self.right_foot_id) * \
                np.in1d(geom1, self.ground_id)).any():
            right_foot_contact = True
        if (qpos[2] < 0.15):
            done_by_early_stop = True
            
        # self.read_sensor_data()

        basequat = self.sim.data.get_body_xquat("head_pan_link")
        quat_desired = Quaternion(array=[1,0,0,0])  
        baseQuatError = (quat_desired.conjugate * Quaternion(array=basequat)).angle

        pelvis_quat = Quaternion(array=qpos[3:7])
        pelvis_vel_local = pelvis_quat.conjugate.rotate(qvel[0:3])

        target_qpos_ordered = np.concatenate([target_data_qpos[0:12], self.init_q_desired[19:27]])

        robot_scale = 30.87/1025.0
        torque_capacity_scale = 5.0/300.0
        mimic_body_orientation_reward =  0.3 * exp(-13.2*abs(baseQuatError)) 
        qpos_regulation = 0.35*exp(-4.0*(np.linalg.norm(target_qpos_ordered - qpos[7:])**2))
        qvel_regulation = 0.05*exp(-0.01*(np.linalg.norm(self.init_qvel[6:] - qvel[6:])**2))
        body_vel_reward = 0.3*exp(-3.0*(np.linalg.norm(pelvis_vel_local[0:2] - self.target_vel)**2))
        contact_force_penalty = 0.1*(exp(-0.0005*(np.linalg.norm(self.ft_left_foot/robot_scale) + np.linalg.norm(self.ft_right_foot/robot_scale))))
        torque_regulation = 0.05*exp(-0.01*(np.linalg.norm(self.action_cur/torque_capacity_scale)))
        torque_diff_regulation = 0.6*(exp(-0.01*(np.linalg.norm((self.action_cur - self.action_last)/torque_capacity_scale))))
        qacc_regulation = 0.05*exp(-20.0*(np.linalg.norm(self.qvel_pre - qvel[6:])**2))
        weight_scale = sum(self.model.body_mass[:]) / sum(self.nominal_body_mass)
        force_ref_reward = 0.1*exp(-0.001*(np.linalg.norm(self.ft_left_foot[2] - robot_scale*weight_scale*self.mocap_data[self.mocap_data_idx,34]))) \
                        + 0.1*exp(-0.001*(np.linalg.norm(self.ft_right_foot[2] - robot_scale*weight_scale*self.mocap_data[self.mocap_data_idx,35])))

        double_support_force_diff_regulation = 0.0
        if ((self.mocap_data_idx < 300) or \
            (3300 < self.mocap_data_idx and self.mocap_data_idx < 3600) or \
            (1500 < self.mocap_data_idx and self.mocap_data_idx < 2100)): # Double support
            if (right_foot_contact and left_foot_contact):
                foot_contact_reward = 0.2
            else:
                foot_contact_reward = 0.0
            double_support_force_diff_regulation = 0.0#0.05*exp(-0.005*(np.linalg.norm(self.ft_right_foot[2] - 0.5 * 9.81 * sum(self.model.body_mass)))) + 0.05*exp(-0.005*(np.linalg.norm(self.ft_left_foot[2] - 0.5 * 9.81 * sum(self.model.body_mass))))
            
        elif (300 < self.mocap_data_idx and self.mocap_data_idx < 1500):
            if (right_foot_contact and not left_foot_contact):
                foot_contact_reward = 0.2
            else:
                foot_contact_reward = 0.0
        else:
            if (not right_foot_contact and left_foot_contact):
                foot_contact_reward = 0.2
            else:
                foot_contact_reward = 0.0
        contact_force_diff_regulation = 0.2*exp(-0.01*(np.linalg.norm((self.ft_left_foot - self.ft_left_foot_pre)/robot_scale) + np.linalg.norm((self.ft_right_foot - self.ft_right_foot_pre)/robot_scale)))
        
        force_thres_penalty = 0.0
        if ((abs(self.ft_left_foot[2]) > 1.4 * 9.81 * sum(self.model.body_mass)) or (abs(self.ft_right_foot[2]) > 1.4 * 9.81 * sum(self.model.body_mass))):
            force_thres_penalty = -0.08
        
        force_diff_thres_penalty = 0.0
        if ((abs(self.ft_left_foot[2] - self.ft_left_foot_pre[2]) >  0.5 * 9.81 * sum(self.model.body_mass)) or (abs(self.ft_right_foot[2] - self.ft_right_foot_pre[2]) > 0.5 * 9.81 * sum(self.model.body_mass))):
            force_diff_thres_penalty = -0.05

        reward = mimic_body_orientation_reward + qpos_regulation + qvel_regulation + contact_force_penalty + torque_regulation + torque_diff_regulation + qacc_regulation + body_vel_reward + foot_contact_reward + contact_force_diff_regulation + double_support_force_diff_regulation + force_thres_penalty + force_diff_thres_penalty + force_ref_reward
        
        self.ft_left_foot_pre = np.copy(self.ft_left_foot)
        self.ft_right_foot_pre = np.copy(self.ft_right_foot)

        # self.data_log.append(np.concatenate([self.ft_left_foot, self.ft_right_foot]))
        
        if not done_by_early_stop:
            self.epi_len += 1
            self.epi_reward += reward
            if (self.spec is not None and self.epi_len == self.spec.max_episode_steps):
                print("Epi len: ", self.epi_len)
                # np.savetxt("./result/"+"action_log"+".txt", self.action_log,delimiter='\t')
                # np.savetxt("./result/"+"data_log_torque_250Hz"+".txt", self.data_log,delimiter='\t')

                return self._get_obs(), reward, done_by_early_stop, dict(episode=dict(r=self.epi_reward, l=self.epi_len), \
                    specific_reward=dict(mimic_body_orientation_reward=mimic_body_orientation_reward,\
                                        qpos_regulation=qpos_regulation,\
                                        qvel_regulation=qvel_regulation,\
                                        contact_force_penalty=contact_force_penalty,
                                        torque_regulation=torque_regulation,
                                        torque_diff_regulation=torque_diff_regulation,
                                        qacc_regulation=qacc_regulation,
                                        body_vel_reward=body_vel_reward,
                                        foot_contact_reward=foot_contact_reward,
                                        contact_force_diff_regulation=contact_force_diff_regulation,
                                        double_support_force_diff_regulation=double_support_force_diff_regulation,
                                        force_thres_penalty=force_thres_penalty,
                                        force_diff_thres_penalty=force_diff_thres_penalty,
                                        force_ref_reward=force_ref_reward))

            return self._get_obs(), reward, done_by_early_stop, \
                dict(specific_reward=dict(mimic_body_orientation_reward=mimic_body_orientation_reward, \
                                        qpos_regulation=qpos_regulation,\
                                        qvel_regulation=qvel_regulation,\
                                        contact_force_penalty=contact_force_penalty,
                                        torque_regulation=torque_regulation,
                                        torque_diff_regulation=torque_diff_regulation,
                                        qacc_regulation=qacc_regulation,
                                        body_vel_reward=body_vel_reward,
                                        foot_contact_reward=foot_contact_reward,
                                        contact_force_diff_regulation=contact_force_diff_regulation,
                                        double_support_force_diff_regulation=double_support_force_diff_regulation,
                                        force_thres_penalty=force_thres_penalty,
                                        force_diff_thres_penalty=force_diff_thres_penalty,
                                        force_ref_reward=force_ref_reward))
        else:
            mimic_body_orientation_reward = 0.0
            qpos_regulation = 0.0
            qvel_regulation = 0.0
            contact_force_penalty = 0.0
            torque_regulation = 0.0
            torque_diff_regulation = 0.0
            qacc_regulation = 0.0
            body_vel_reward = 0.0
            foot_contact_reward = 0.0
            contact_force_diff_regulation = 0.0
            double_support_force_diff_regulation = 0.0
            force_thres_penalty = 0.0
            force_diff_thres_penalty = 0.0
            force_ref_reward = 0.0
            reward = 0.0

            print("Epi len: ", self.epi_len)            
            # try: os.mkdir('./result')
            # except: pass
            # np.savetxt("./result/"+"data_log"+".txt", self.data_log,delimiter='\t')
            # np.savetxt("./result/"+"action_log"+".txt", self.action_log,delimiter='\t')

            return self._get_obs(), reward, done_by_early_stop, dict(episode=dict(r=self.epi_reward, l=self.epi_len),\
                 specific_reward=dict(mimic_body_orientation_reward=mimic_body_orientation_reward, \
                                    qpos_regulation=qpos_regulation,\
                                    qvel_regulation=qvel_regulation,\
                                    contact_force_penalty=contact_force_penalty,
                                    torque_regulation=torque_regulation,
                                    torque_diff_regulation=torque_diff_regulation,
                                    qacc_regulation=qacc_regulation,
                                    body_vel_reward=body_vel_reward,
                                    foot_contact_reward=foot_contact_reward,
                                    contact_force_diff_regulation=contact_force_diff_regulation,
                                    double_support_force_diff_regulation=double_support_force_diff_regulation,
                                    force_thres_penalty=force_thres_penalty,
                                    force_diff_thres_penalty=force_diff_thres_penalty,
                                    force_ref_reward=force_ref_reward))

    def reset_model(self):
        self.time = 0.0
        self.epi_len = 0
        self.epi_reward = 0

        # width = 100
        # height = 100
        # img = np.random.uniform(0.0, 1.0 ,(width, height))
        # img[49:51,49:51] = 0.0
        # flat_idx = np.random.uniform(0.0, 1.0, height)
        # img[:,flat_idx>0.5] = 0.0
        # img = img.reshape(1,-1)
        # self.model.hfield_data[:] = img

        # Dynamics Randomization
        body_mass = np.array(self.nominal_body_mass)
        body_mass_noise = np.random.uniform(0.6, 1.4, len(body_mass))
        body_mass = body_mass * body_mass_noise
        self.model.body_mass[:]  = body_mass
        
        body_inertia = np.array(self.nominal_body_inertia)
        body_inertia_noise = np.random.uniform(0.6, 1.4, len(body_inertia))
        body_inertia = np.multiply(body_inertia, body_inertia_noise[:, np.newaxis])
        self.model.body_inertia[:]  = body_inertia
        

        body_ipos = np.array(self.nominal_body_ipos)
        body_ipos_noise = np.random.uniform(0.6, 1.4, len(body_ipos))
        body_ipos = np.multiply(body_ipos, body_ipos_noise[:, np.newaxis])
        self.model.body_ipos[:]  = body_ipos
        
        dof_damping = np.array(self.nominal_dof_damping)
        dof_damping_noise = np.random.uniform(0.6, 1.4, len(dof_damping))#np.random.uniform(1/noise_scale, noise_scale, len(dof_damping))
        dof_damping = dof_damping * dof_damping_noise
        self.model.dof_damping[:]  = dof_damping

        dof_frictionloss = np.array(self.nominal_dof_frictionloss)
        dof_frictionloss_noise = np.random.uniform(0.6, 1.4, len(dof_frictionloss))#np.random.uniform(1/noise_scale, noise_scale, len(dof_frictionloss))
        dof_frictionloss = dof_frictionloss * dof_frictionloss_noise 
        self.model.dof_frictionloss[:]  = dof_frictionloss

        # Motor Constant Randomization
        self.motor_constant_scale = np.random.uniform(0.70, 1.30, 12)
        # self.motor_constant_scale = np.tile(motor_constant_scale, 2)

        # Delay Randomization
        self.action_delay = np.random.randint(low=5, high=12)

        if (np.random.rand(1) < 0.5):
            self.init_mocap_data_idx = 0#np.random.randint(low=0, high=self.mocap_data_num)
        else:
            self.init_mocap_data_idx = 1800
        init_q_pos = np.copy(self.init_q_desired)
        # init_q_pos[19:] = self.mocap_data[self.init_mocap_data_idx,13:]
        
        init_qvel = np.copy(self.init_qvel)
        # init_qvel[0] = np.random.uniform(0.0, 0.3)
        self.set_state(init_q_pos, init_qvel)  

        # init_q_pos[2] = init_q_pos[2] - (self.sim.data.get_body_xpos("R_Foot_Link")[2] - 0.15811) # To offset so that feet are on ground
        # self.set_state(init_q_pos, self.init_qvel)  

        self.qpos_noise = init_q_pos[7:]
        self.qpos_pre = init_q_pos[7:]
        self.qvel_noise.fill(0)
        self.qvel_lpf.fill(0)

        self.read_sensor_data()
        self.ft_left_foot_pre = np.copy(self.ft_left_foot)
        self.ft_right_foot_pre = np.copy(self.ft_right_foot)

        self.target_vel = np.array([np.random.uniform(-0.2, 0.5), np.random.uniform(-0.2, 0.2)])
        
        self.q_bias = np.random.uniform(-3.14/100.0, 3.14/100.0, 12)
        self.quat_bias = np.random.uniform(-3.14/150.0, 3.14/150.0, 3)

        self.action_log = []
        self.data_log = []

        self.action_last.fill(0)
        self.qvel_pre.fill(0)

        self.perturb_timing = np.random.randint(1,control_freq_scale*2000)
        self.obs_buf = []
        self.action_buf = []
        self.action_raw.fill(0.0)

        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 1.0
        self.viewer.cam.lookat[2] = 2.0
        self.viewer.cam.elevation = -20

    def read_sensor_data(self):
        self.ft_left_foot = self.data.sensordata[self.ft_left_foot_adr:self.ft_left_foot_adr+3]
        self.ft_right_foot = self.data.sensordata[self.ft_right_foot_adr:self.ft_right_foot_adr+3]

    def perturbation_start(self):
        self.perturbation_on = True