import os 
import torch

act_project_dir = os.getenv("ACT_PROJECT_DIR")
assert act_project_dir is not None, "No ACT_PROJECT_DIR set! Make sure you have ran the source setup_env.sh!"

# data directory
DATA_DIR = act_project_dir + '/data/'
# checkpoint directory
CHECKPOINT_DIR = act_project_dir + '/checkpoints/'

device = 'cpu'
if torch.cuda.is_available(): device = 'cuda'
os.environ['DEVICE'] = device

PANDA_TASK_CONFIG = {
    'dataset_dir': DATA_DIR + 'latch/',
    'episode_len': 300,
    'state_dim': 8,
    'action_dim': 8,
    'cam_width': 640,
    'cam_height': 480,
    'camera_names': ['front'],
    'camera_port': 0
}

# policy config
PANDA_POLICY_CONFIG = {
    'lr': 1e-5,
    'num_queries': 30,
    'kl_weight': 10,
    'hidden_dim': 512,
    'dim_feedforward': 3200,
    'lr_backbone': 1e-5,
    'backbone': 'resnet18',
    'enc_layers': 4,
    'dec_layers': 7,
    'nheads': 8,
    'camera_names': ['front'],
    'policy_class': 'ACT',
    'temporal_agg': True,
    'state_dim': PANDA_TASK_CONFIG['state_dim'],
    'action_dim': PANDA_TASK_CONFIG['action_dim'],
}

# training config
PANDA_TRAIN_CONFIG = {
    'seed': 42,
    'num_epochs': 500000,
    'batch_size_val': 8,
    'batch_size_train': 8,
    'eval_ckpt_name': 'policy_epoch_90000_seed_42.ckpt',
    'checkpoint_dir': CHECKPOINT_DIR,
    'wandb': True
}
