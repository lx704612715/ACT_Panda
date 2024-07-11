import os 
import yaml
import argparse
import numpy as np
from tqdm import tqdm
from loguru import logger

from pytransform3d.rotations import active_matrix_from_extrinsic_euler_xyz
import matplotlib.pyplot as plt
from pytransform3d.transformations import plot_transform
from pytransform3d.camera import plot_camera

from act_panda.demonstration.record_episodes import ImgTrajectoryRecorder
from contact_lfd.LfDusingEC.utils.utils_simple_lfd import generate_uniform_distributed_demo_poses
from contact_lfd.LfDusingEC.utils.utils_robotMath import RtToTrans
from contact_lfd.LfDIP.controller.arm_controller import Arm_Controller


def plot_camera_poses(center_point, poses):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Define camera intrinsic parameters with larger sensor size and focal length
    focal_length = 0.004  # Increased to 40mm
    sensor_width = 0.0072  # Doubled to 72mm
    sensor_height = 0.0048  # Doubled to 48mm
    M = np.array([[focal_length, 0, sensor_width / 2],
                  [0, focal_length, sensor_height / 2],
                  [0, 0, 1]])

    # Set plot limits
    ax.set_xlim(center_point[0]-0.1, center_point[0]+0.1)
    ax.set_ylim(center_point[1]-0.1, center_point[1]+0.1)
    ax.set_zlim(0.0, 0.5)

    # Plot the center of the circle as the object
    ax.plot(center_point[0], center_point[1], center_point[2], 'ro')

    for pose in poses:
        # Plot each pose as a camera
        plot_camera(ax, M=M, cam2world=pose, virtual_image_distance=0.01, sensor_size=(sensor_width, sensor_height))

    plt.show()


if __name__ == "__main__":
    
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

    center_point = np.array([0.550187, 0.104202, 0])
    init_R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    init_base_ht_ee = RtToTrans(R=init_R, t=center_point)
    init_q = np.array([0.00440534, -0.157901, 0.199459, -2.00603, -0.0158633, 1.89083, 0.987942])
    arm_ctrl.move_to_joint_position(init_q, completion_time=5)

    poses_1 = generate_uniform_distributed_demo_poses(num_poses=12, radius=0.05, height=0.2)
    poses_2 = generate_uniform_distributed_demo_poses(num_poses=18, radius=0.1, height=0.3)
    # poses_3 = generate_uniform_distributed_demo_poses(num_poses=13, radius=0.10, height=0.35)

    poses = init_base_ht_ee @ np.vstack([poses_1, poses_2])

    plot_camera_poses(center_point, poses)

    # this is used to prevent stop recording in the middle
    start_idx = 24

    for i in range(start_idx, len(poses)):
        pose = poses[i]
        # get the IK solution
        dq = arm_ctrl.get_ik_solution(arm_ctrl.q_queue[-1], pose)
        logger.info("Moving to the {}th starting pose".format(i))

        if dq is None:
            logger.warning("Cartesian Position Control!")
            arm_ctrl.move_to_carte_position(pose, completion_time=6)
        else:
            if np.any(dq > arm_ctrl.q_lim_ub) or np.any(dq < arm_ctrl.q_lim_lb):
                logger.warning("Exceeding Joint Limits!")
                dq = np.clip(dq, arm_ctrl.q_lim_lb, arm_ctrl.q_lim_ub)
            arm_ctrl.move_to_joint_position(dq, completion_time=6)

        before_record_q = arm_ctrl.q_queue[-1]
        logger.info("Start {}th Recording".format(i))
        arm_ctrl.activate_gravity_com()
        recorder.recording()
        arm_ctrl.activate_gravity_com()
        input("Back to Init Pose")
        arm_ctrl.move_to_joint_position(dq, completion_time=6)

        # finish the recording
        recorder.reset_recording()
        logger.info("Recording Finished".format(i))

