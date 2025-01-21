#!/usr/bin/python3
import copy
import os
import h5py
import yaml
import argparse
import numpy as np
import message_filters
from collections import deque
from pynput.keyboard import Listener, Key

# Own scripts
from contact_lfd.LfDIP.controller.arm_controller import Arm_Controller
from contact_lfd.LfDusingEC.utils.utils_simple_lfd import TrajectoryRecorder
from contact_lfd.LfDusingEC.utils.utils_robotMath import compute_distance_between_two_transforms

# ROS
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class ImgTrajectoryRecorder(TrajectoryRecorder):
    def __init__(self,  arm_ctrl: Arm_Controller, data_path, record=True, demo_idx=0, min_trans=0.005, min_rot=0.1, config=None):
        super().__init__(arm_ctrl, data_path, record, demo_idx, min_trans, min_rot)
        self.curt_iH_rgb_img = None
        self.curt_static_rgb_img = None
        self.cfg = config
        self.camera_name = "front"
        self.iH_rgb_imgs = []
        self.static_rgb_imgs = []
        self.q_traj = []
        self.dq_traj = []
        self.ee_ft_robot_traj = []
        self.base_ht_ee_traj = []
        self.gripper_status = []
        self.switch_gripper_indexes = []
        self.num_waypoints = 0

        self.iH_img_deque = deque(maxlen=1)
        self.static_img_deque = deque(maxlen=1)
        self.bridge = CvBridge()

        # in Hand Camera
        rospy.wait_for_message('/camera/color/image_raw', Image)
        self.iH_image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.iH_img_cb, queue_size=1)

        # static camera
        rospy.wait_for_message('/ptu_camera/color/image_raw', Image)
        self.static_image_sub = rospy.Subscriber('/ptu_camera/color/image_raw', Image, self.static_img_cb, queue_size=1)

    def iH_img_cb(self, iH_img_msg):
        self.curt_iH_rgb_img = self.bridge.imgmsg_to_cv2(iH_img_msg, desired_encoding="rgb8")
        self.iH_img_deque.append(self.curt_iH_rgb_img)
        
    def static_img_cb(self, static_img_msg):
        self.curt_static_rgb_img = self.bridge.imgmsg_to_cv2(static_img_msg, desired_encoding="rgb8")
        self.static_img_deque.append(self.curt_static_rgb_img)

    def recording(self, **kwargs):
        self.recorder_logger.debug("Press Enter to Start Recording")
        input()
        self.recorder_logger.critical("press ESC to stop recording, press Enter to grasp object")
        self.listener.start()

        last_base_ht_ee = self.arm_ctrl.get_curt_cartesian_pose()
        last_gripper_width = self.arm_ctrl.q_with_gripper_queue[-1][-1]

        self.base_ht_ee_traj.append(last_base_ht_ee)
        self.iH_rgb_imgs.append(self.curt_iH_rgb_img)
        self.static_rgb_imgs.append(self.curt_static_rgb_img)
        self.q_traj.append(self.arm_ctrl.q_with_gripper_queue[-1])
        self.ee_ft_robot_traj.append(self.arm_ctrl.ft_robot_queue[-1])

        # we use the difference between gripper width as the gripper velocity
        dq_with_gripper = np.append(self.arm_ctrl.dq_queue[-1], 0)
        self.dq_traj.append(dq_with_gripper) 
        self.gripper_status.append(int(self.arm_ctrl.gripper_ctl.is_grasping()))

        while self.start_recording and not rospy.is_shutdown() and self.arm_ctrl.ham_working:
            curt_base_ht_ee = self.arm_ctrl.get_curt_cartesian_pose()
            curt_gripper_width = self.arm_ctrl.q_with_gripper_queue[-1][-1]
            diff_gripper_width = curt_gripper_width - last_gripper_width
            trans, rot = compute_distance_between_two_transforms(curt_base_ht_ee, last_base_ht_ee)
            rospy.loginfo_throttle(1, "Recording!")
            if trans >= self.min_trans or rot >= self.min_rot or abs(diff_gripper_width) >= 0.01:
                if len(self.iH_img_deque) > 0 and len(self.static_img_deque) > 0:
                    self.base_ht_ee_traj.append(curt_base_ht_ee)
                    self.ee_ft_robot_traj.append(self.arm_ctrl.ft_robot_queue[-1])
                    self.gripper_status.append(int(self.arm_ctrl.gripper_ctl.is_grasping()))
                    self.q_traj.append(self.arm_ctrl.q_with_gripper_queue[-1])
                    dq_with_gripper = np.append(self.arm_ctrl.dq_queue[-1], diff_gripper_width)
                    self.dq_traj.append(dq_with_gripper)
                    self.iH_rgb_imgs.append(self.iH_img_deque.pop())
                    self.static_rgb_imgs.append(self.static_img_deque.pop())

                    last_base_ht_ee = np.copy(curt_base_ht_ee)
                    last_gripper_width = np.copy(curt_gripper_width)

            rospy.loginfo_throttle(1, "Recording is Going On!!")

        # add the final pose
        rospy.sleep(1)
        self.iH_rgb_imgs.append(self.iH_img_deque.pop())
        self.static_rgb_imgs.append(self.static_img_deque.pop())
        self.q_traj.append(self.arm_ctrl.q_with_gripper_queue[-1])
        self.ee_ft_robot_traj.append(self.arm_ctrl.ft_robot_queue[-1])
        self.base_ht_ee_traj.append(self.arm_ctrl.get_curt_cartesian_pose())
        self.gripper_status.append(int(self.arm_ctrl.gripper_ctl.is_grasping()))

        dq_with_gripper = np.append(self.arm_ctrl.dq_queue[-1], 0)
        self.dq_traj.append(dq_with_gripper)

        self.num_waypoints = len(self.q_traj)
        self.listener.stop()
        self.recorder_logger.warning("Stop Recording for Corrective Demonstration")

        # construct states and actions
        action = copy.deepcopy(self.q_traj[1:])
        action.append(self.q_traj[-1])

        # construct the dataset
        self.data = {'/observations/qpos': self.q_traj,
                     '/observations/qvel': self.dq_traj,
                     '/observations/images/front': self.iH_rgb_imgs,
                     '/observations/images/static': self.static_rgb_imgs,
                     '/observations/ee_ft_robot': self.ee_ft_robot_traj,
                     '/observations/base_ht_ee': self.base_ht_ee_traj,
                     '/action': action}

        self.convert_data_to_hdf5()

    def convert_data_to_hdf5(self):
        max_timesteps = len(self.data['/observations/qpos'])
        # create data dir if it doesn't exist
        os.makedirs(self.data_path, exist_ok=True)

        # count number of files in the directory
        idx = len([name for name in os.listdir(self.data_path) if os.path.isfile(os.path.join(self.data_path, name))])
        dataset_path = os.path.join(self.data_path, f'episode_{idx}')

        # save the data
        with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
            root.attrs['sim'] = True
            obs = root.create_group('observations')
            qpos = obs.create_dataset('qpos', (max_timesteps, self.cfg['state_dim']))
            qvel = obs.create_dataset('qvel', (max_timesteps, self.cfg['state_dim']))
            ee_ft_robot = obs.create_dataset('ee_ft_robot', (max_timesteps, 6))
            base_ht_ee = obs.create_dataset('base_ht_ee', (max_timesteps, 4, 4))
            action = root.create_dataset('action', (max_timesteps, self.cfg['action_dim']))
            image = obs.create_group('images')
            image.create_dataset("front", (max_timesteps, self.cfg['cam_height'], self.cfg['cam_width'], 3), dtype='uint8',
                                 chunks=(1, self.cfg['cam_height'], self.cfg['cam_width'], 3))
            image.create_dataset("static", (max_timesteps, self.cfg['cam_height'], self.cfg['cam_width'], 3), dtype='uint8',
                                 chunks=(1, self.cfg['cam_height'], self.cfg['cam_width'], 3))
            for name, array in self.data.items():
                root[name][...] = array

        self.recorder_logger.info("Exported dataset to: {}".format(dataset_path + '.hdf5'))
    
    def reset_recording(self):
        """ reset the data to prepare for the next recording step
        """
        self.data = {}
        self.start_recording = True
        self.iH_rgb_imgs = []
        self.static_rgb_imgs = []
        self.q_traj = []
        self.dq_traj = []
        self.base_ht_ee_traj = []
        self.ee_ft_robot_traj = []
        self.gripper_status = []
        self.switch_gripper_indexes = []
        self.num_waypoints = 0
        self.listener = Listener(on_press=self._on_press)
        self.recorder_logger.critical("Reset Recording Finished!")


if __name__ == "__main__":
    # parse the task name via command line
    rospy.init_node("training")
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='puzzle')  # FB: FurnitureBench
    parser.add_argument('--config', type=str, default='train_config_act_puzzle')
    args = parser.parse_args()
    config_name = args.config
    act_project_dir = os.getenv("ACT_PROJECT_DIR")
    config_path = act_project_dir + '/act_panda/config/' + config_name + '.yaml'
    config = yaml.load(open(config_path, "r"), Loader=yaml.Loader)

    task_name = args.task
    task_config = config['task_config']
    dataset_dir = act_project_dir + '/data/' + task_name
    os.makedirs(dataset_dir, exist_ok=True)

    arm_ctrl = Arm_Controller(gripper=True)
    arm_ctrl.activate_gravity_com()

    recorder = ImgTrajectoryRecorder(arm_ctrl, dataset_dir, min_trans=0.005, min_rot=0.05, config=task_config)
    recorder.recording()
    arm_ctrl.opening_gripper()

    print("debug")


