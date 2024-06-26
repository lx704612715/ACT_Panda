from config.config import PANDA_TASK_CONFIG
import copy
import os
import h5py
import argparse
import numpy as np
from collections import deque

# Own scripts
from contact_lfd.LfDIP.controller.arm_controller import Arm_Controller
from contact_lfd.LfDusingEC.utils.utils_simple_lfd import TrajectoryRecorder
from contact_lfd.LfDusingEC.utils.utils_robotMath import compute_distance_between_two_transforms

# ROS
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class ImgTrajectoryRecorder(TrajectoryRecorder):
    def __init__(self,  arm_ctrl: Arm_Controller, data_path, record=True, demo_idx=0, min_trans=0.005, min_rot=0.1, config=PANDA_TASK_CONFIG):
        super().__init__(arm_ctrl, data_path, record, demo_idx, min_trans, min_rot)
        self.curt_rgb_img = None
        self.cfg = config
        self.camera_name = "front"
        self.rgb_imgs = []
        self.q_traj = []
        self.dq_traj = []
        self.base_ht_ee_traj = []
        self.gripper_status = []
        self.switch_gripper_indexes = []
        self.num_waypoints = 0

        self.img_deque = deque(maxlen=1)
        self.bridge = CvBridge()

        ros_rgb_topic = '/camera/color/image_raw'
        rospy.wait_for_message(ros_rgb_topic, Image)
        self.color_image_sub = rospy.Subscriber(ros_rgb_topic, Image, self.img_cb, queue_size=10)

    def img_cb(self, color_img_msg):
        self.curt_rgb_img = self.bridge.imgmsg_to_cv2(color_img_msg, desired_encoding="rgb8")
        self.img_deque.append(self.curt_rgb_img)
    
    def recording(self, **kwargs):
        self.recorder_logger.debug("Press Enter to Start Recording")
        input()
        self.recorder_logger.critical("press ESC to stop recording, press Enter to grasp object")
        self.listener.start()

        last_base_ht_ee = self.arm_ctrl.base_ht_ee_queue[-1]
        last_gripper_width = self.arm_ctrl.q_with_gripper_queue[-1][-1]

        self.rgb_imgs.append(self.curt_rgb_img)
        self.q_traj.append(self.arm_ctrl.q_with_gripper_queue[-1])
        # we use the difference between gripper width as the gripper velocity
        dq_with_gripper = np.append(self.arm_ctrl.dq_queue[-1], 0)
        self.dq_traj.append(dq_with_gripper) 
        self.base_ht_ee_traj.append(last_base_ht_ee)
        self.gripper_status.append(int(self.arm_ctrl.gripper_ctl.is_grasping()))

        while self.start_recording and not rospy.is_shutdown() and self.arm_ctrl.ham_working:
            curt_base_ht_ee = self.arm_ctrl.base_ht_ee_queue[-1]
            curt_gripper_width = self.arm_ctrl.q_with_gripper_queue[-1][-1]
            diff_gripper_width = curt_gripper_width - last_gripper_width
            trans, rot = compute_distance_between_two_transforms(curt_base_ht_ee, last_base_ht_ee)
            rospy.loginfo_throttle(1, "Recording!")
            if trans >= self.min_trans or rot >= self.min_rot or abs(diff_gripper_width) >= 0.01:
                if len(self.img_deque) > 0:
                    self.base_ht_ee_traj.append(curt_base_ht_ee)
                    self.gripper_status.append(int(self.arm_ctrl.gripper_ctl.is_grasping()))
                    self.q_traj.append(self.arm_ctrl.q_with_gripper_queue[-1])
                    dq_with_gripper = np.append(self.arm_ctrl.dq_queue[-1], diff_gripper_width)
                    self.dq_traj.append(dq_with_gripper)
                    self.rgb_imgs.append(self.img_deque.pop())

                    last_base_ht_ee = np.copy(curt_base_ht_ee)
                    last_gripper_width = np.copy(curt_gripper_width)

            rospy.loginfo_throttle(1, "Recording is Going On!!")

        # add the final pose
        rospy.sleep(1)
        self.rgb_imgs.append(self.img_deque.pop())
        self.q_traj.append(self.arm_ctrl.q_with_gripper_queue[-1])
        dq_with_gripper = np.append(self.arm_ctrl.dq_queue[-1], 0)
        self.dq_traj.append(dq_with_gripper) 
        self.base_ht_ee_traj.append(self.arm_ctrl.base_ht_ee_queue[-1])
        self.gripper_status.append(int(self.arm_ctrl.gripper_ctl.is_grasping()))
        self.num_waypoints = len(self.q_traj)

        self.listener.stop()
        self.recorder_logger.warning("Stop Recording for Corrective Demonstration")

        # construct states and actions
        action = copy.deepcopy(self.q_traj[1:])
        action.append(self.q_traj[-1])

        # construct the dataset
        self.data = {'/observations/qpos': self.q_traj,
                     '/observations/qvel': self.dq_traj,
                     '/observations/images/front': self.rgb_imgs,
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
            action = root.create_dataset('action', (max_timesteps, cfg['action_dim']))
            image = obs.create_group('images')
            image.create_dataset("front", (max_timesteps, self.cfg['cam_height'], self.cfg['cam_width'], 3), dtype='uint8',
                                 chunks=(1, self.cfg['cam_height'], self.cfg['cam_width'], 3))
            for name, array in self.data.items():
                root[name][...] = array

        self.recorder_logger.info("Exported dataset to: {}".format(dataset_path + '.hdf5'))


if __name__ == "__main__":
    # parse the task name via command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='latch')
    parser.add_argument('--num_episodes', type=int, default=0)
    args = parser.parse_args()
    task = args.task
    num_episodes = args.num_episodes
    cfg = PANDA_TASK_CONFIG

    arm_ctrl = Arm_Controller(gripper=True)
    arm_ctrl.activate_gravity_com()
    recorder = ImgTrajectoryRecorder(arm_ctrl, cfg['dataset_dir']+task, min_trans=0.005, min_rot=0.05, config=PANDA_TASK_CONFIG)

    recorder.recording()
    arm_ctrl.opening_gripper()

    print("debug")


