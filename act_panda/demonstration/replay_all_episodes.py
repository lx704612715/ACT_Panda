from act_panda.config.config import PANDA_POLICY_CONFIG, PANDA_TASK_CONFIG, PANDA_TRAIN_CONFIG # must import first
import os
from tqdm import tqdm
import mediapy as media
import matplotlib.pyplot as plt
from act_panda.utils.utils import load_hdf5
from contact_lfd.LfDusingEC.vis.base_plot_funcs import plot_multi_lines

act_project_dir = os.getenv("ACT_PROJECT_DIR")
DATA_DIR = act_project_dir + '/data/'
# Given a dataset dir, iterate all episode in the dir and align the data shape
dataset_name = 'insertion_puzzle_aligned'

dataset_dir = DATA_DIR + dataset_name + '/'
output_vis_dir = DATA_DIR + dataset_name + '_vis' + '/'
all_episodes_names = [name for name in os.listdir(dataset_dir) if os.path.isfile(os.path.join(dataset_dir, name))]
os.makedirs(output_vis_dir, exist_ok=True)

shapes = []
aligned_data = dict()
# first get all dataset shape
for name in tqdm(all_episodes_names):
    aligned_data[name] = dict()
    dataset_path = os.path.join(dataset_dir, name)
    qpos, qvel, action, image_dict = load_hdf5(dataset_path=dataset_path)

    shapes.append(qpos.shape[0])
    aligned_data[name]['qpos'] = qpos
    aligned_data[name]['qvel'] = qvel
    aligned_data[name]['action'] = action
    aligned_data[name]['image_dict'] = image_dict

    # export videos
    video_path = output_vis_dir + name.split('.')[0] + '_front.mp4'
    media.write_video(video_path, image_dict['front'], fps=30)

    video_path = output_vis_dir + name.split('.')[0] + '_static.mp4'
    media.write_video(video_path, image_dict['static'], fps=30)

    # export plots
    fig_path = output_vis_dir + name.split('.')[0] + '.png'
    plot_multi_lines(Y=qpos, title="joint positions", legend=['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8'], x_label="step")
    plt.savefig(fig_path)