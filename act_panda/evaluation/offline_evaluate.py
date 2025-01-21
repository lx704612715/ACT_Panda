# standard library
import tqdm
import yaml
import pickle
import argparse
import torchvision.transforms as T
from loguru import logger

# Deep Learning Scripts
from act_panda.utils.utils import *
from contact_lfd.LfDusingEC.vis.base_plot_funcs import plot_multi_lines, generate_fig_for_multi_lines



if __name__ == "__main__":
    act_project_dir = os.getenv("ACT_PROJECT_DIR")
    # load dataset
    parser = argparse.ArgumentParser()
    # parser.add_argument('--config', type=str, default='train_config_act_puzzle')
    parser.add_argument('--config', type=str, default='train_config_diffusion_puzzle')
    args = parser.parse_args()
    config_name = args.config
    config_path = act_project_dir + '/act_panda/config/' + config_name + '.yaml'
    config = yaml.load(open(config_path, "r"), Loader=yaml.Loader)

    task_cfg = config['task_config']
    train_cfg = config['train_config']
    policy_cfg = config['policy_config']
    
    ckpt_dir = act_project_dir + train_cfg['eval_ckpt_dir']
    ckpt_path = ckpt_dir + train_cfg['eval_ckpt_name']
    dataset_dir = act_project_dir + task_cfg['dataset_dir']

    output_vis_dir = act_project_dir + '/data/' + config_name + "_eval/"
    # load dataset
    os.makedirs(output_vis_dir, exist_ok=True)
    all_episodes_names = [name for name in os.listdir(dataset_dir) if os.path.isfile(os.path.join(dataset_dir, name))]

    # Load normalization stats
    stats_path = ckpt_dir + 'dataset_stats.pkl'
    stats = pickle.load(open(stats_path, 'rb'))

    # load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    policy = make_policy(policy_cfg['policy_class'], policy_cfg)
    loading_status = policy.load_state_dict(torch.load(ckpt_path, map_location=torch.device(device)))
    policy.to(device)
    policy.eval()

    # number of training episodes
    data_dir = act_project_dir + task_cfg['dataset_dir']
    num_episodes = len(os.listdir(data_dir))

    # hyperparameter
    k = 0.01
    num_queries = policy_cfg['num_queries']
    total_diff = []

    for episode_name in all_episodes_names:
        episode_data_path = dataset_dir + episode_name
        qpos, qvel, action, image_dict = load_hdf5(dataset_path=episode_data_path)
        iH_imgs = image_dict['front']
        static_imgs = image_dict['static']

        num_waypoints = qpos.shape[0]
        preprocess = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        all_time_actions = np.zeros([task_cfg['episode_len'], task_cfg['episode_len'] + num_queries, task_cfg['state_dim']])
        diff = []
        hat_action_list = []
        gt_action_list = []
        for i in tqdm.tqdm(range(num_waypoints-1)):
            obs_iH_img = iH_imgs[i]
            obs_static_img = static_imgs[i]
            obs_q = qpos[i]

            norm_q_pos = (obs_q - stats['qpos_mean']) / stats['qpos_std']
            norm_q_pos_tensor = torch.from_numpy(norm_q_pos).float().to(device).unsqueeze(0)
            iH_img_tensor = preprocess(obs_iH_img).to(device).unsqueeze(0)
            static_img_tensor = preprocess(obs_static_img).to(device).unsqueeze(0)
            img_tensor = torch.vstack([iH_img_tensor, static_img_tensor]).unsqueeze(0)  # we only use one camera
            all_actions_tensor = policy(norm_q_pos_tensor, img_tensor)
            all_actions_array = all_actions_tensor.cpu().detach().numpy()

            # perform temporal aggregation (Trunking)
            if policy_cfg["policy_class"] == "ACT":
                all_time_actions[[i], i:i+num_queries] = all_actions_array
                actions_for_curr_step = all_time_actions[:, i]
                actions_populated = np.all(actions_for_curr_step != 0, axis=1)
                actions_for_curr_step = actions_for_curr_step[actions_populated]
                exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                exp_weights = exp_weights / exp_weights.sum()
                exp_weights = exp_weights.reshape(-1, 1)
                raw_action = np.sum(actions_for_curr_step * exp_weights, axis=0, keepdims=True)
            else:
                raw_action = all_actions_array[0][0]

            # denormalize actions
            hat_action = raw_action * stats['action_std'] + stats['action_mean']
            # compare with ground truth
            gt_action = qpos[i+1]
            # diff
            gt_action_list.append(gt_action)
            hat_action_list.append(hat_action)
            diff.append(np.abs(hat_action - gt_action))

        # compute overall statistic
        diff_array = np.vstack(diff)
        gt_action_array = np.vstack(gt_action_list)
        hat_action_array = np.vstack(hat_action_list)
        total_diff.append(diff_array)

        fig, axarr = generate_fig_for_multi_lines(n_rows=3, n_cols=1, figsize=(10, 8))
        y_axis_max = np.max(gt_action_array-gt_action_array[0])
        y_axis_min = np.min(gt_action_array-gt_action_array[0])
        # plot gt
        plot_multi_lines(Y=gt_action_array-gt_action_array[0], ax=axarr[0], title="Ground Truth", x_label="Timestep", y_label="Joint Values",
                         plot_legend=True, legend=['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8'], linewidth=2, grid=True)
        axarr[0].set_ylim([y_axis_min, y_axis_max])
        # plot hat_action
        plot_multi_lines(Y=hat_action_array-gt_action_array[0], ax=axarr[1], title="Predictions", x_label="Timestep", y_label="Joint Values",
                         plot_legend=True, legend=['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8'], linewidth=2, grid=True)
        axarr[1].set_ylim([y_axis_min, y_axis_max])
        plot_multi_lines(Y=diff_array, ax=axarr[2], title="Difference", x_label="Timestep", y_label="Joint Values",
                         plot_legend=True, legend=['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8'], linewidth=2, grid=True)
        # safe the plot to the folder with the name of epsiode
        output_path = output_vis_dir + episode_name.split(".")[0] + '_eval.png'
        fig.tight_layout()
        fig.savefig(output_path)

    # compute overall loss
    total_diff = np.mean(np.vstack(total_diff))
    logger.debug("Total Difference: {}".format(total_diff, axis=0))
    with open(output_vis_dir + "total_loss.txt", "w") as file:
        file.write(str(total_diff))


        





