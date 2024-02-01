from collections import OrderedDict
import os
from gym import error, spaces
from gym.utils import seeding
import numpy as np
from os import path
import gym

import random
import time

from numpy.core.arrayprint import format_float_scientific

try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))


class TocabiEnv(gym.envs.mujoco.MujocoEnv):
    """Superclass for all MuJoCo environments.
    """
    def __init__(self, model_path, frame_skip):

        if model_path.startswith("/"):
            fullpath = model_path
        else:
            # assets 폴더에 있는 .xml 파일로 경로 지정
            fullpath = os.path.join(os.path.dirname(__file__), "assets", model_path)
        if not path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)
        self.frame_skip = frame_skip
        # 무조코.py 모델 로드
        # dyros_tocabi.xml 파일 << 시뮬 모델 파일
            # 일단 간단하게 요약하면
            # 1. 메쉬 경로 설정, 마찰 설정, 중력 옵션설정
            # 2. 메쉬파일(STL), 이름 설정
            ## HK : sacle 변수는 뭔가?scale="0.001 0.001 0.001" 예상 : xyz 1/1000배 크기 scaling
            # 2-1. texture, material
            # 이해못함, 환경일듯
            # 3. base_link body 로부터 Tree 구조로 링크-조인트-링크 선언
            # pos size quat type
            # inertia mass pos
            # joint axis damping armature:전기자관성(로터,기어), frictionloss, range
            # 4. actuator motor ctrlrange 설정
            # 5. FT 센서 설정 (force, torque) 
        self.model = mujoco_py.load_model_from_path(fullpath)
        self.sim = mujoco_py.MjSim(self.model)
        self.data = self.sim.data
        self.viewer = None
        self._viewers = {}

        self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()

        # Custum variable
        self.done_init = False
        self.epi_len = 0
        self.epi_reward = 0
        self.ground_collision_check_id = []
        self.self_collision_check_id = []
        self.ground_id = []
        self.left_foot_id = []
        self.right_foot_id = []
        self.deep_mimic_env = True
        self.time = 0.0
        
        # self.init_q_desited[?] 
        # HK: [?] 38번까지 있는 이유를? 모르겠음.. 예상 : 버추얼 조인트 6개 추가? 33+6 = 39dof라서 인덱스가 0~38?   
        self.init_q_desired = np.copy(self.data.qpos)
        self.init_q_desired[7:] = 0.0
        self.init_q_desired[9] = -0.24
        self.init_q_desired[10] = 0.6
        self.init_q_desired[11] = -0.36
        self.init_q_desired[15] = -0.24
        self.init_q_desired[16] = 0.6
        self.init_q_desired[17] = -0.36
        self.init_q_desired[22] = 0.3
        self.init_q_desired[23] = 0.3
        self.init_q_desired[24] = 1.5
        self.init_q_desired[25] = -1.27
        self.init_q_desired[26] = -1.0
        self.init_q_desired[28] = -1.0
        self.init_q_desired[32] = -0.3
        self.init_q_desired[33] = -0.3
        self.init_q_desired[34] = -1.5
        self.init_q_desired[35] = 1.27
        self.init_q_desired[36] = 1.0
        self.init_q_desired[38] = 1.0
        self.set_state(self.init_q_desired, self.init_qvel,)
       
        # 엥 ? target_position 에서 self.init_q_desired[0:3] 을 쓰는데?
        self.target_position = np.copy(self.init_q_desired[0:3])

        self.nominal_body_mass = np.copy(self.model.body_mass)
        self.nominal_body_inertia = np.copy(self.model.body_inertia)
        self.nominal_body_ipos = np.copy(self.model.body_ipos)
        self.nominal_dof_damping = np.copy(self.model.dof_damping)
        # firction loss란?
        # 마찰로 인해 손실되는 동력(에너지/토크(?))
        self.nominal_dof_frictionloss = np.copy(self.model.dof_frictionloss)

        self.dof_damping = np.array(self.nominal_dof_damping)
        self.dof_frictionloss = np.array(self.nominal_dof_frictionloss)
        #질량 노이즈
        self.body_mass_noise = np.random.uniform(0.8, 1.2, len(self.nominal_body_mass))

        # 모터 상수
        motor_constant_scale = np.random.uniform(0.95, 1.05, 6)
        # np.tile(x, n) : x를 n번 쌓음
        # 즉 모터상수 스케일이 12차원이됨
        self.motor_constant_scale = np.tile(motor_constant_scale, 2)

        # Deep Mimic
        # 딥미믹 https://xbpeng.github.io/projects/DeepMimic/index.html
        # Example-guided DRL
        self.init_mocap_data_idx = 0
        self.mocap_data_idx = 0
        self.mocap_data = np.genfromtxt('motions/processed_data_tocabi_walk.txt', encoding='ascii')
        self.mocap_data_num = len(self.mocap_data) - 1
        self.mocap_cycle_dt = 0.0005

        #np.zeros_like(A) : 안에있는 A array or matrix or tuple 을 같은 모양으로된 zeros 반환
        #노이즈를 위한 변수
        self.qpos_noise = np.zeros_like(self.sim.data.qpos[7:])
        self.qvel_noise = np.zeros_like(self.sim.data.qvel[6:])
        #lpf를 위한 변수
        self.qvel_lpf = np.zeros_like(self.sim.data.qvel[6:])
        #이전값저장
        self.qpos_pre = np.zeros_like(self.sim.data.qpos[7:])
        self.qvel_pre = np.zeros_like(self.sim.data.qvel[6:])

        #목표 속도, 시작속도, 마지막속도 를 v_x = 0.0~0.8로 랜덤화
        self.target_vel = np.array([np.random.uniform(0.0, 0.8), 0.0])        
        self.start_target_vel = np.array([np.random.uniform(0.0, 0.8), 0.0])
        self.final_target_vel = np.array([np.random.uniform(0.0, 0.8), 0.0])
        # HK: 어디쓰지 ?
        self.vel_change_duration = 0
        self.cur_vel_change_duration = 0
        
        #ft 센서를 위한 변수 초기화
        self.ft_left_foot = np.zeros(3)
        self.ft_left_foot_pre = np.zeros(3)
        self.ft_right_foot = np.zeros(3)
        self.ft_right_foot_pre = np.zeros(3)
        self.torque_left_foot = np.zeros(3)
        self.torque_right_foot = np.zeros(3)
        #FT센서 값 저장?
        for sensor_name, id in self.sim.model._sensor_name2id.items():
            if sensor_name == "LF_Force_sensor":
                    self.ft_left_foot_adr = self.sim.model.sensor_adr[id]
            elif sensor_name == "LF_Torque_sensor":
                    self.torque_left_foot_adr = self.sim.model.sensor_adr[id]
            elif sensor_name == "RF_Force_sensor":
                    self.ft_right_foot_adr = self.sim.model.sensor_adr[id]
            elif sensor_name == "RF_Torque_sensor":
                    self.torque_right_foot_adr = self.sim.model.sensor_adr[id]
        
        self.action_log = []
        self.data_log = []
        # HK: 액션 딜레이가 뭐지?
        self.action_delay = 5

        # 히스토리, 스킵
        # observation HK: 스킵하는 이유?
        self.num_obs_hist = 5
        self.num_obs_skip = 2
        self.obs_buf = []
        self.action_buf = []

        #bias 랜덤 생성.
        #np.random.uniform(a,b,n) : a(최소값)부터 b(최대값)까지 값을 균등 분포로 n개 뽑는다.
        #np.random.normal() : 정규분포
        # 하체 q값
        self.q_bias = np.random.uniform(-0.034, 0.034, 12)
        # 쿼터니안 바이어스? HK : 어디에 쓰는거지? quaternian이면 4개여야하는데 w빼고 뽑는건가?(x,y,z만?)
        self.quat_bias = np.random.uniform(-3.14/300.0, 3.14/300.0, 3)
        #ft센서 bias인데, HK: 2개만 뽑는 이유? 단위? 예상 : x,y만 다루는 학습?
        self.ft_bias = np.random.uniform(-100.0, 100.0, 2)
        #마찬가지로 모멘트 bias. HK:모멘트는 mx면 x축기준 모멘트 두개를 뽑는거? my면??? (잘모르겠다)
        self.ft_mx_bias = np.random.uniform(-10.0, 10.0, 2)
        self.ft_my_bias = np.random.uniform(-30.0, 30.0, 2)

        # HK: ??뭔지잘모르겠음 둘다 37개의 데이터가 있음. 평균값?
        # np.genfromtxt 는 텍스트파일로부터 그 값을이용, array를 생성해줌
        self.obs_mean = np.genfromtxt('data/obs_mean_fixed.txt', encoding='ascii')
        self.obs_var = np.genfromtxt('data/obs_variance_fixed.txt', encoding='ascii')

        #안씀
        self.metadata = {
            'render.modes': ['human', 'rgb_array', 'depth_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        
        #액션스페이스 플래그
        self.custom_action_space = True
        self._set_action_space() # 어디쓰는거지?. 아래에 함수, action_space 생성에 사용됨.

        #액션 샘플링
        action = self.action_space.sample()
        self.action_last = self.action_space.sample()[0:-1]
        self.action_cur = np.copy(self.action_last)
        self.action_raw = self.action_space.sample()
        self.action_raw_pre = np.copy(self.action_raw)

        # HK: high라고 이름붙인이유? 예상 : high frequency? high, 최대값?
        _, self.actuator_high = self.model.actuator_ctrlrange.copy().T
        # np.concatenate() : 위아래로 배열 합침
        self.action_high = np.concatenate([self.actuator_high[0:12], [self.dt]])

        if self.deep_mimic_env:
            action = np.zeros_like(action)

        # tocabi_walk.py의 step 함수
        observation, _reward, done, _info = self.step(action)
        assert not done

        self._set_observation_space(observation)

        self.seed()

    def _set_action_space(self):
        if self.custom_action_space:
            # HK: dyros_tocabi.xml의 actuator_ctrlrange를 불러오는게 맞는지? [0:12]가 하체이긴함.
            # HK : 근데이게 스케일이 뭐지? degree? current??
            bounds = self.model.actuator_ctrlrange.copy()[0:12]
            bounds[:] = [-1.0, 1.0]
            # 하체 액추에이터 12차원 -1.0~1.0 + [[0.0, 1.0]] 
            # HK: 뒤에 두개는 뭐지? 아!!! 지지발?
            bounds = np.concatenate([bounds, [[0.0, 1.0]]])
            low, high = bounds.T
            #gym action space, Box
            #https://codetorial.net/articles/cartpole/space.html
            self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        else:
            bounds = self.model.actuator_ctrlrange.copy()
            low, high = bounds.T
            self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        return self.action_space