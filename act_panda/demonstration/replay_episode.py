import os
import h5py
import mediapy as media
import matplotlib.pyplot as plt
from act_panda.utils.utils import load_hdf5

from config.config import TASK_CONFIG, ROBOT_PORTS

# play cam video
data_file = 'data/sort/episode_0.hdf5'
# data_file = 'data/demo/trained.hdf5'
qpos, qvel, action, image_dict = load_hdf5(dataset_path=data_file)
images = image_dict['front']
media.show_video(images, fps=30)

media.write_video('v.mp4', images, fps=30)

for cam_name, image_list in image_dict.items():
    media.show_video(image_list, fps=30)

# plot qpos
plt.figure()
plt.plot(qpos, label=list(range(1, 6)))
plt.xlabel('step')
plt.ylabel('angle [rad]')
plt.title('joint positions')
plt.legend()