import os 
import yaml
import rospy
import argparse
import numpy as np
from loguru import logger

from act_panda.demonstration.record_episodes import ImgTrajectoryRecorder
from contact_lfd.LfDusingEC.utils.utils_simple_lfd import generate_uniform_distributed_demo_poses, plot_camera_poses
from contact_lfd.LfDusingEC.utils.utils_robotMath import RtToTrans
from contact_lfd.LfDIP.controller.arm_controller import Arm_Controller



if __name__ == "__main__":
    rospy.init_node('record_uniform_demonstration', anonymous=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='insertion_puzzle')
    parser.add_argument('--config', type=str, default='train_config_act')
    args = parser.parse_args()
    config_name = args.config
    act_project_dir = os.getenv("ACT_PROJECT_DIR")
    config_path = act_project_dir + '/act_panda/config/' + config_name + '.yaml'
    config = yaml.load(open(config_path, "r"), Loader=yaml.Loader)
    
    task_name = args.task
    task_config = config['task_config']
    dataset_dir = act_project_dir + '/act_panda/data/' + task_name
    os.makedirs(dataset_dir, exist_ok=True)

    arm_ctrl = Arm_Controller(gripper=True)
    recorder = ImgTrajectoryRecorder(arm_ctrl, dataset_dir, min_trans=0.005, min_rot=0.05, config=task_config)

    center_point = np.array([0.6932, 0.0752, 0.0])
    init_R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    init_base_ht_ee = RtToTrans(R=init_R, t=center_point)
    init_q = np.array([0.00440534, -0.157901, 0.199459, -2.00603, -0.0158633, 1.89083, 0.987942])
    arm_ctrl.move_to_joint_position(init_q, completion_time=5)

    poses_1 = generate_uniform_distributed_demo_poses(num_poses=12, radius=0.05, height=0.2, start_deg=0, end_deg=np.deg2rad(180))
    poses_2 = generate_uniform_distributed_demo_poses(num_poses=18, radius=0.1, height=0.3, start_deg=0, end_deg=np.deg2rad(180))
    # poses_3 = generate_uniform_distributed_demo_poses(num_poses=13, radius=0.10, height=0.35)

    poses = init_base_ht_ee @ np.vstack([poses_1, poses_2])

    plot_camera_poses(center_point, poses)

    # this is used to prevent stop recording in the middle
    start_idx = 10

    for i in range(start_idx, len(poses)):
        pose = poses[i]
        # get the IK solution
        dq = arm_ctrl.get_ik_solution(d_base_ht_ee=pose, curt_q=arm_ctrl.q_queue[-1])
        logger.info("Moving to the {}th starting pose".format(i))

        if dq is None:
            logger.warning("Cartesian Position Control!")
            arm_ctrl.move_to_carte_position(pose, completion_time=2)
        else:
            if np.any(dq > arm_ctrl.q_lim_ub) or np.any(dq < arm_ctrl.q_lim_lb):
                logger.warning("Exceeding Joint Limits!")
                dq = np.clip(dq, arm_ctrl.q_lim_lb, arm_ctrl.q_lim_ub)
            arm_ctrl.move_to_joint_position(dq, completion_time=2)

        before_record_q = arm_ctrl.q_queue[-1]
        logger.info("Start {}th Recording".format(i))
        arm_ctrl.activate_gravity_com()
        recorder.recording()
        arm_ctrl.activate_gravity_com()
        input("Back to Init Pose")
        arm_ctrl.move_to_joint_position(dq, completion_time=2)

        # finish the recording
        recorder.reset_recording()
        logger.info("Recording Finished".format(i))

