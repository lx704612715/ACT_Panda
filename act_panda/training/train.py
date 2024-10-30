import os
import sys
import yaml
import wandb
import pickle
import argparse
import matplotlib.pyplot as plt
act_project_dir = os.getenv("ACT_PROJECT_DIR")
sys.path.append(act_project_dir)

from act_panda.config.config import PANDA_POLICY_CONFIG, PANDA_TASK_CONFIG, PANDA_TRAIN_CONFIG # must import first
# act_project_dir = os.getenv("ACT_PROJECT_DIR") + '/act_panda'

from loguru import logger
from copy import deepcopy
from datetime import datetime

from tqdm import tqdm
from act_panda.utils.utils import *


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    # save training curves
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        plt.plot(np.linspace(0, num_epochs-1, len(train_history)), train_values, label='train')
        plt.plot(np.linspace(0, num_epochs-1, len(validation_history)), val_values, label='validation')
        # plt.ylim([-0.1, 1])
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
    # print(f'Saved plots to {ckpt_dir}')


def train_policy(train_dataloader, val_dataloader, policy_config, train_cfg):
    # load policy
    policy = make_policy(policy_config['policy_class'], policy_config)
    policy.to('cuda')

    # load optimizer
    optimizer = make_optimizer(policy_config['policy_class'], policy)

    # create checkpoint dir if not exists
    os.makedirs(checkpoint_dir, exist_ok=True)

    train_history = []
    validation_history = []
    best_checkpoints = []
    min_val_loss = np.inf

    for epoch in tqdm(range(train_cfg['num_epochs']), file=sys.stdout):
        # validation
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            for batch_idx, data in enumerate(val_dataloader):
                forward_dict = forward_pass(data, policy)
                epoch_dicts.append(forward_dict)
            val_epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(val_epoch_summary)

            epoch_val_loss = val_epoch_summary['loss']
            
            # Update best checkpoints list and manage files
            if len(best_checkpoints) < 5 or epoch_val_loss < max(best_checkpoints, key=lambda x: x[1])[1]:
                if len(best_checkpoints) == 5:
                    # Remove the worst checkpoint's file
                    worst_checkpoint = max(best_checkpoints, key=lambda x: x[1])
                    os.remove(os.path.join(checkpoint_dir, f'best_ckpt_epoch_{worst_checkpoint[0]}.ckpt'))
                    best_checkpoints.remove(worst_checkpoint)
                
                # Add the new best checkpoint
                best_checkpoints.append((epoch, epoch_val_loss, deepcopy(policy.state_dict())))
                best_checkpoints.sort(key=lambda x: x[1])  # Sort checkpoints by validation loss

                # Save the new checkpoint file
                ckpt_path = os.path.join(checkpoint_dir, f'best_ckpt_epoch_{epoch}.ckpt')
                torch.save(policy.state_dict(), ckpt_path)
            
        summary_string = ''
        for k, v in val_epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '

        tqdm.write(f'Validation Summary - Epoch {epoch}: Loss: {val_epoch_summary["loss"]:.4f}')

        # training
        policy.train()
        optimizer.zero_grad()
        for batch_idx, data in enumerate(train_dataloader):
            forward_dict = forward_pass(data, policy)
            # backward
            loss = forward_dict['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))
        train_epoch_summary = compute_dict_mean(train_history[(batch_idx+1)*epoch:(batch_idx+1)*(epoch+1)])
        epoch_train_loss = train_epoch_summary['loss']
        # logger.info("Train Loss".format(epoch_train_loss))
        summary_string = ''
        for k, v in train_epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '

        tqdm.write(f'Training Summary - Epoch {epoch}: Loss: {train_epoch_summary["loss"]:.4f}')

        if epoch % 50 == 0 and train_cfg['wandb']:
            wandb.log({f"train/{k}": v for k, v in train_epoch_summary.items()})
            wandb.log({f"val/{k}": v for k, v in val_epoch_summary.items()})

        if epoch % 1000 == 0:
            plot_history(train_history, validation_history, epoch, checkpoint_dir, train_cfg['seed'])

    ckpt_path = os.path.join(checkpoint_dir, f'policy_last.ckpt')
    torch.save(policy.state_dict(), ckpt_path)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='FB')
    parser.add_argument('--config', type=str, default='train_config_act_FB')
    args = parser.parse_args()
    config_name = args.config
    act_project_dir = os.getenv("ACT_PROJECT_DIR")
    config_path = act_project_dir + '/act_panda/config/' + config_name + '.yaml'
    config = yaml.load(open(config_path, "r"), Loader=yaml.Loader)

    task_cfg = config['task_config']
    train_cfg = config['train_config']
    policy_config = config['policy_config']

    # parse the task name via command line
    task = args.task
    
    # Get the current date and time
    now = datetime.now()
    # Construct the subdirectory name
    subdir_name = f"{task}_{now.strftime('%H-%M')}_{now.strftime('%m-%d')}"

    # configs
    checkpoint_dir = os.path.join(act_project_dir+train_cfg['checkpoint_dir'], subdir_name)
    logger.info("Checkpoint dir: {}".format(checkpoint_dir))

    # device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # wandb
    if train_cfg["wandb"]:
        wandb.init(project="act-panda", name=subdir_name)
        wandb.run.name = subdir_name

    # set seed
    set_seed(train_cfg['seed'])
    # create ckpt dir if not exists
    os.makedirs(checkpoint_dir, exist_ok=True)

    # number of training episodes
    data_dir = act_project_dir + task_cfg['dataset_dir']
    logger.info("Dataset dir: {}".format(data_dir))

    num_episodes = len(os.listdir(data_dir))

    # load data
    train_dataloader, val_dataloader, stats, _ = load_data(data_dir, num_episodes, task_cfg['camera_names'],
                                                           train_cfg['batch_size_train'], train_cfg['batch_size_val'])
    # save stats
    stats_path = os.path.join(checkpoint_dir, f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    # train
    train_policy(train_dataloader, val_dataloader, policy_config, train_cfg)