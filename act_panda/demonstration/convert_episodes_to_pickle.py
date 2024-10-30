from act_panda.config.config import PANDA_POLICY_CONFIG, PANDA_TASK_CONFIG, PANDA_TRAIN_CONFIG  # must import first
import os
import h5py
import numpy as np
import pandas as pd
from loguru import logger
from act_panda.utils.utils import load_hdf5

if __name__ == "__main__":

    act_project_dir = os.getenv("ACT_PROJECT_DIR")
    DATA_DIR = act_project_dir + '/data/'
    # Given a dataset dir, iterate all episode in the dir and align the data shape
    dataset_dir = DATA_DIR + 'insertion_puzzle/'

    all_episodes_names = [name for name in os.listdir(dataset_dir) if os.path.isfile(os.path.join(dataset_dir, name))]

    shapes = []
    aligned_data = dict()
    # first get all dataset shape
    for name in all_episodes_names:
        aligned_data[name] = dict()
        dataset_path = os.path.join(dataset_dir, name)
        qpos, qvel, action, image_dict = load_hdf5(dataset_path=dataset_path)

        shapes.append(qpos.shape[0])
        aligned_data[name]['qpos'] = qpos
        aligned_data[name]['qvel'] = qvel
        aligned_data[name]['action'] = action
        aligned_data[name]['image_dict'] = image_dict

    # mean_episode_len = int(np.mean(shapes))
    max_episode_len = int(np.max(shapes))
    logger.info(f"Mean episode length: {max_episode_len}")
    input("Press Enter to start aligning...")
    max_episode_len = 160  # the episode length should be able to divided by action_horizon.....

    data_annotation_frame = pd.DataFrame()

    # new overwrite all datasets to make sure every episode has the same length
    for name in all_episodes_names:
        dataset_path = os.path.join(dataset_dir, name)

        with h5py.File(dataset_path, 'r+') as root:

            del root['observations/qpos']
            root.create_dataset('observations/qpos', data=aligned_data[name]['qpos'])

            del root['observations/qvel']
            root.create_dataset('observations/qvel', data=aligned_data[name]['qvel'])

            del root['action']
            root.create_dataset('action', data=aligned_data[name]['action'])

            for cam_name in root[f'/observations/images/'].keys():
                del root[f'/observations/images/{cam_name}']
                root.create_dataset(f'/observations/images/{cam_name}', data=aligned_data[name]['image_dict'][cam_name])

    logger.info("Alignment is Finished!")