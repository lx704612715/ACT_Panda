from act_panda.config.config import PANDA_POLICY_CONFIG, PANDA_TASK_CONFIG, PANDA_TRAIN_CONFIG  # must import first
# standard library
import os
import yaml
import tqdm
import rospkg
import torch
import numpy as np
import pandas as pd
import pytransform3d.transformations as py3d_trans
import pickle
import torchvision.models as models
from collections import deque
from PIL import Image as PIL_Image
from scipy.special import softmax

# ROS
import tf
import rospy
import message_filters
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import torchvision.transforms as T
from contact_lfd.LfDusingEC.utils.utils_ros import get_BB_goal, HT_to_pq

# Scripts
from contact_lfd.LfDusingEC.utils.utils_ros import get_BB_Joint_goal
from contact_lfd.LfDIP.controller.arm_controller import Arm_Controller

# Deep Learning Scripts
from act_panda.utils.utils import *


class ACT_Controller(Arm_Controller):
    """ this controller is used for reproducing of non-contact demonstrated skills
        to instantiate this controller, we assign a sub-set of human demonstration, this controller needs to know:
            1. should the robot grasp at the end of the servoing phase?
            2. the path for the trained model
    """
    def __init__(self, gripper, experiment_dir):
        super().__init__(gripper=gripper)
        self.experiment_dir = experiment_dir

        self.bridge = CvBridge()
        self.synchronized_robot_images = deque(maxlen=1)

        # setup message synchronizer
        ros_rgb_topic = '/camera/color/image_raw'
        rospy.wait_for_message(ros_rgb_topic, Image)
        self.color_image_sub = rospy.Subscriber(ros_rgb_topic, Image, self.img_cb, queue_size=10)

        # ACT models
        self.device = torch.device("cuda")
        self.preprocess = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        
        # load model
        self.stats = None  # mean and std for each dataset
        self.policy = None
        self.train_cfg = PANDA_TRAIN_CONFIG
        self.policy_cfg = PANDA_POLICY_CONFIG
        self.task_cfg = PANDA_TASK_CONFIG
        self.load_model()

        # control loop
        self.init_base_ht_ee = None

    def img_cb(self, color_img_msg):
        curt_rgb_img = self.bridge.imgmsg_to_cv2(color_img_msg, desired_encoding="rgb8")
        curt_robot_q_state = self.q_with_gripper_queue[-1]
        self.synchronized_robot_images.append([curt_robot_q_state, curt_rgb_img])

    def load_model(self):
        ckpt_dir = "/act_panda/training/checkpoints/latch_backup/"
        ckpt_path = ckpt_dir + "policy_epoch_15000_seed_42.ckpt"
        # ckpt_path = os.path.join(self.train_cfg['checkpoint_dir'], self.train_cfg['eval_ckpt_name'])
        self.policy = make_policy(self.policy_cfg['policy_class'], self.policy_cfg)
        loading_status = self.policy.load_state_dict(torch.load(ckpt_path, map_location=torch.device(self.device)))
        self.ctrl_logger.info("Loading status {}".format(loading_status))
        self.policy.to(self.device)
        self.policy.eval()

        # Load normalization stats
        self.ctrl_logger.info('Loaded'.format(ckpt_path))

        stats_path = ckpt_dir + 'dataset_stats.pkl'
        with open(stats_path, 'rb') as f:
            self.stats = pickle.load(f)

    def execution(self):
        kp = np.array([500, 400, 400, 300, 300, 200, 100])
        kv = np.sqrt(kp)
        self.activate_blackboard_joint_ha(kp=kp, kv=kv, interpolation_type="linear", reinterpolation="1", epsilon="1")

        obs_replay = []
        action_replay = []
        t = 0
        query_frequency = 1
        completion_time = 0.1
        num_queries = self.policy_cfg['num_queries']
        all_time_actions = torch.zeros([self.task_cfg['episode_len'], self.task_cfg['episode_len'] + num_queries,
                                        self.task_cfg['state_dim']]).to(self.device)
        qpos_history = torch.zeros((1, self.task_cfg['episode_len'], self.task_cfg['state_dim'])).to(self.device)

        while not rospy.is_shutdown() and self.ham_working:
            if len(self.synchronized_robot_images) == 0:
                continue

            q_pos, rgb_image = self.synchronized_robot_images.popleft()
            norm_q_pos = (q_pos - self.stats['qpos_mean']) / self.stats['qpos_std']
            norm_q_pos_tensor = torch.from_numpy(norm_q_pos).float().to(self.device).unsqueeze(0)
            qpos_history[:, t] = norm_q_pos_tensor
            img_tensor = self.preprocess(rgb_image).to(self.device).unsqueeze(0)
            img_tensor = img_tensor.unsqueeze(0) # we only use one camera

            all_actions = self.policy(norm_q_pos_tensor, img_tensor)

            if self.policy_cfg['temporal_agg']:
                all_time_actions[[t], t:t + num_queries] = all_actions
                actions_for_curr_step = all_time_actions[:, t]
                actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                actions_for_curr_step = actions_for_curr_step[actions_populated]
                k = 0.01
                exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                exp_weights = exp_weights / exp_weights.sum()
                exp_weights = torch.from_numpy(exp_weights.astype(np.float32)).to(self.device).unsqueeze(dim=1)
                raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
            else:
                raw_action = all_actions[:, t % query_frequency]


            raw_action = raw_action.cpu().detach().numpy()[0]
            action = raw_action * self.stats['action_std'] + self.stats['action_mean']

            d_q_pos = action[:7]
            clip_d_q_pos = self.clip_joint_velocity(q_pos[:7], d_q_pos, completion_time)
            joint_goal = get_BB_Joint_goal(kp=kp, kv=kv, d_q=clip_d_q_pos, completion_time=completion_time)
            self.joint_goal_pub.publish(joint_goal)
            rospy.sleep(completion_time)

            if d_q_pos[-1] <= 0.02:
                self.grasping()

        self.ctrl_logger.critical("Done")

    def clip_joint_velocity(self, curt_q, d_q, completion_time):
        max_diff_q = np.array([0.1*completion_time for i in range(7)])  # read from franka specifications
        diff_q = d_q - curt_q
        clip_diff_q = np.clip(diff_q, -max_diff_q, max_diff_q)
        clip_d_q = curt_q + clip_diff_q
        return clip_d_q


def main():
    # load config file
    act_controller = ACT_Controller(gripper=True, experiment_dir=None)
    act_controller.execution()
    return 0


if __name__ == "__main__":
    main()