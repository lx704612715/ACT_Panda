from act_panda.config.config import PANDA_POLICY_CONFIG, PANDA_TASK_CONFIG, PANDA_TRAIN_CONFIG # must import first
import os
import h5py
import numpy as np
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

    for name in aligned_data.keys():
        episode_len = aligned_data[name]['qpos'].shape[0]
        if episode_len > max_episode_len:
            aligned_data[name]['qpos'] = aligned_data[name]['qpos'][:max_episode_len]
            aligned_data[name]['qvel'] = aligned_data[name]['qvel'][:max_episode_len]
            aligned_data[name]['action'] = aligned_data[name]['action'][:max_episode_len]
            for cam_name in aligned_data[name]['image_dict'].keys():
                aligned_data[name]['image_dict'][cam_name] = aligned_data[name]['image_dict'][cam_name][:mean_episode_len]
        else:
            diff_episode_len = max_episode_len - episode_len
            # repeat the last data value {diff_episode_len} times
            aligned_data[name]['qpos'] = np.vstack([aligned_data[name]['qpos'], np.repeat(aligned_data[name]['qpos'][-1:], diff_episode_len, axis=0)])
            aligned_data[name]['qvel'] = np.vstack([aligned_data[name]['qvel'], np.repeat(aligned_data[name]['qvel'][-1:], diff_episode_len, axis=0)])
            aligned_data[name]['action'] = np.vstack([aligned_data[name]['action'], np.repeat(aligned_data[name]['action'][-1:], diff_episode_len, axis=0)])
            for cam_name in aligned_data[name]['image_dict'].keys():
                aligned_data[name]['image_dict'][cam_name] = np.vstack([aligned_data[name]['image_dict'][cam_name], np.repeat(aligned_data[name]['image_dict'][cam_name][-1:], diff_episode_len, axis=0)])

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