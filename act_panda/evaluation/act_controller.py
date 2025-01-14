from act_panda.config.config import PANDA_POLICY_CONFIG, PANDA_TASK_CONFIG, PANDA_TRAIN_CONFIG  # must import first
# standard library
import os
import yaml
import rospy
import pickle
import argparse
import message_filters
import torchvision.transforms as T
from collections import deque
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

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
    def __init__(self, config, gripper):
        super().__init__(gripper=gripper)
        self.config = config
        self.train_cfg = self.config['train_config']
        self.policy_cfg = self.config['policy_config']
        self.task_cfg = self.config["task_config"]

        self.bridge = CvBridge()
        self.synchronized_robot_images = deque(maxlen=1)

        # setup message synchronizer
        rospy.wait_for_message('/camera/color/image_raw', Image)
        self.iH_image_sub = message_filters.Subscriber('/camera/color/image_raw', Image, queue_size=10)
        # static camera
        rospy.wait_for_message('/ptu_camera/color/image_raw', Image)
        self.static_image_sub = message_filters.Subscriber('/ptu_camera/color/image_raw', Image, queue_size=10)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.iH_image_sub, self.static_image_sub],
                                                              queue_size=10, slop=0.1)
        self.ts.registerCallback(self.img_cb)

        # ACT models
        self.device = torch.device("cuda")
        self.preprocess = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        
        # load model
        self.stats = None  # mean and std for each dataset
        self.policy = None

        # control loop
        self.init_base_ht_ee = None

    def img_cb(self, iH_color_img_msg, static_color_img_msg):
        iH_color_img = self.bridge.imgmsg_to_cv2(iH_color_img_msg, desired_encoding="rgb8")
        static_color_img = self.bridge.imgmsg_to_cv2(static_color_img_msg, desired_encoding="rgb8")
        curt_robot_q_state = self.q_with_gripper_queue[-1]
        self.synchronized_robot_images.append([curt_robot_q_state, iH_color_img, static_color_img])

    def load_model(self, ckpt_dir, ckpt_path):
        self.ctrl_logger.info('Loaded'.format(ckpt_dir))

        self.policy = make_policy(self.policy_cfg['policy_class'], self.policy_cfg)
        loading_status = self.policy.load_state_dict(torch.load(ckpt_path, map_location=torch.device(self.device)))
        self.ctrl_logger.info("Loading status {}".format(loading_status))
        self.policy.to(self.device)
        self.policy.eval()

        # Load normalization stats
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

        all_time_actions = np.zeros([self.task_cfg['episode_len'], self.task_cfg['episode_len'] + num_queries, self.task_cfg['state_dim']])
        qpos_history = np.zeros((1, self.task_cfg['episode_len'], self.task_cfg['state_dim']))

        # we use mock state as the robot state for the gripper width, because we cannot control the panda gripper contiously
        mock_last_d_gripper_width = self.q_with_gripper_queue[-1][-1]

        # we assume the panda robot has grasped the puzzle piece
        mock_last_d_gripper_width = self.stats['example_qpos'][0][-1]

        while not rospy.is_shutdown() and self.ham_working:
            if len(self.synchronized_robot_images) == 0:
                continue

            q_pos, iH_color_img, static_color_img = self.synchronized_robot_images.popleft()
            q_pos[-1] = mock_last_d_gripper_width
            norm_q_pos = (q_pos - self.stats['qpos_mean']) / self.stats['qpos_std']
            # replace the gripper width with the last desired gripper width

            norm_q_pos_tensor = torch.from_numpy(norm_q_pos).float().to(self.device).unsqueeze(0)
            qpos_history[:, t] = norm_q_pos
            iH_img_tensor = self.preprocess(iH_color_img).to(self.device).unsqueeze(0)
            static_img_tensor = self.preprocess(static_color_img).to(self.device).unsqueeze(0)
            img_tensor = torch.vstack([iH_img_tensor, static_img_tensor]).unsqueeze(0) # we only use one camera

            all_actions_tensor = self.policy(norm_q_pos_tensor, img_tensor)
            all_actions_array = all_actions_tensor.cpu().detach().numpy()

            if self.policy_cfg['temporal_agg']:
                all_time_actions[[t], t:t + num_queries] = all_actions_array
                actions_for_curr_step = all_time_actions[:, t]
                actions_populated = np.all(actions_for_curr_step != 0, axis=1)
                actions_for_curr_step = actions_for_curr_step[actions_populated]
                k = 0.01
                exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                exp_weights = exp_weights / exp_weights.sum()
                exp_weights = exp_weights.reshape(-1, 1)
                raw_action = np.sum(actions_for_curr_step * exp_weights, axis=0, keepdims=True)

            else:
                raw_action = all_actions_array[:, t % query_frequency]

            action = raw_action * self.stats['action_std'] + self.stats['action_mean']
            action = action.squeeze()
            d_q_pos = action[:7]

            clip_d_q_pos = self.clip_joint_velocity(q_pos[:7], d_q_pos, completion_time)
            joint_goal = get_BB_Joint_goal(kp=kp, kv=kv, d_q=clip_d_q_pos, completion_time=completion_time)
            self.joint_goal_pub.publish(joint_goal)
            rospy.sleep(completion_time)

            if action[-1] <= 0.03 and not self.gripper_ctl.is_grasping():
                # we assume the robot already grasps the object
                pass
                # self.grasping()

        self.ctrl_logger.critical("Done")

    def clip_joint_velocity(self, curt_q, d_q, completion_time):
        max_diff_q = np.array([0.1*completion_time for i in range(7)])  # read from franka specifications
        diff_q = d_q - curt_q
        clip_diff_q = np.clip(diff_q, -max_diff_q, max_diff_q)
        clip_d_q = curt_q + clip_diff_q
        return clip_d_q


if __name__ == "__main__":
    rospy.init_node('act_controller')
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='eva_config_act')
    args = parser.parse_args()
    config_name = args.config
    act_project_dir = os.getenv("ACT_PROJECT_DIR")
    config_path = act_project_dir + '/act_panda/config/' + config_name + '.yaml'
    config = yaml.load(open(config_path, "r"), Loader=yaml.Loader)

    act_controller = ACT_Controller(config=config, gripper=True)

    ckpt_dir = act_project_dir + config['train_config']['checkpoint_dir']
    ckpt_path = ckpt_dir + config['train_config']['eval_ckpt_name'] 
    
    act_controller.load_model(ckpt_dir, ckpt_path)
    act_controller.execution()
