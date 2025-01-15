""" Evaluation of ACT/Diffusion Policy for Insertion Puzzle

Example:

.. _Google Python Style Guide:
   https://google.github.io/styleguide/pyguide.html

@Author: Xing Li
@Date: 28.Jan.2024
"""

import yaml
import argparse
import random
import copy

# EC state
from contact_lfd.CLfD.planner.ec_planner import ECPlanner
from contact_lfd.CLfD.nodes.ece_demo_processer import ECEDemoProcessor
from act_panda.evaluation.act_controller import ACT_Controller

# kinematic joints
from contact_lfd.LfDusingEC.utils.utils_simple_lfd import *
from contact_lfd.LfDusingEC.utils.utils_robotMath import *
from contact_lfd.CLfD.nodes.rosbag_recorder import RosbagRecorder
from contact_lfd.LfDIP.controller.arm_controller import Arm_Controller
from contact_lfd.LfDusingEC.utils.utils_io import wait_for_human_input


def generate_eval_trajectories():
    path = "/home/lx/experiments/lx/exp_corrective/3101InsertionPuzzle/puzzle_1/eval_augmentation/"
    gt_data = np.load(os.path.join(path, "augmented_img_proc_ref.npy"), allow_pickle=True).item()
    gt_q = gt_data['q']
    gt_base_ht_ee = gt_data['base_ht_ee']

    tmp_base_ht_ee = copy.deepcopy(gt_base_ht_ee)
    tmp_base_ht_ee[2, 3] = 0  # set the z position as 0
    curt_height = gt_base_ht_ee[2, 3]
    executed_height = curt_height + 0.15
    # how to define height?
    ee_ht_ee_traj = generate_uniform_distributed_demo_poses(num_poses=10, radius=0.1, height=executed_height,
                                                            start_deg=np.deg2rad(180), end_deg=np.deg2rad(360))
    d_base_ht_ee = tmp_base_ht_ee @ ee_ht_ee_traj
    # plot_camera_poses(tmp_base_ht_ee[:3, 3], d_base_ht_ee)
    return d_base_ht_ee


if __name__ == "__main__":
    rospy.init_node('act_controller')
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='eva_config_act')
    parser.add_argument('--record', default="1")
    args = parser.parse_args()
    config_name = args.config
    act_project_dir = os.getenv("ACT_PROJECT_DIR")
    config_path = act_project_dir + '/act_panda/config/' + config_name + '.yaml'
    config = yaml.load(open(config_path, "r"), Loader=yaml.Loader)

    act_controller = ACT_Controller(config=config, gripper=True)
    ckpt_dir = act_project_dir + config['train_config']['checkpoint_dir']
    ckpt_path = ckpt_dir + config['train_config']['eval_ckpt_name']
    act_controller.load_model(ckpt_dir, ckpt_path)

    args = parser.parse_args()

    if args.record == "1":
        recorder = RosbagRecorder()
    else:
        recorder = None

    # start key monitor for data recording
    human_cmd = "continue"
    
    center_point = np.array([0.6932, 0.0752, 0.0])
    init_R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    init_base_ht_ee = RtToTrans(R=init_R, t=center_point)
    poses_1 = generate_uniform_distributed_demo_poses(num_poses=12, radius=0.05, height=0.05, start_deg=0, end_deg=np.deg2rad(180))
    poses_2 = generate_uniform_distributed_demo_poses(num_poses=18, radius=0.1, height=0.3, start_deg=0, end_deg=np.deg2rad(180))
    poses = init_base_ht_ee @ np.vstack([poses_1, poses_2])

    counter = 0
    while human_cmd == "continue":

        random_idx = random.randint(12, 29)
        act_controller.move_to_carte_position(carte_position=poses[random_idx], completion_time=4)

        if args.record == "1":
            recorder.set_file_name(file_name="green_trial_" + str(counter))
            recorder.start_recording()

        # run policy execution
        act_controller.execution()

        if args.record == "1":
            recorder.stop_recording()

        logger.info("Finished! Gravity Compensation start!")
        usr_command = wait_for_human_input(desired_input=["c", "s"])
        if "s" in usr_command:
            human_cmd = "stop"

        logger.info("Current Step is {}".format(counter))
        counter += 1
