import yaml

yaml_content = """
# All model checkpoints

task_config:
  dataset_dir: &dataset_dir DATA_DIR + 'latch/'
  episode_len: &episode_len 300
  state_dim: &state_dim 8
  action_dim: &action_dim 8
  cam_width: 640
  cam_height: 480
  camera_names: [ 'front' ]
  camera_port: 0

policy_config:
  policy_class: 'ACT'
  camera_names: *camera_names
  lr: 1e-5
  num_queries: 30
  kl_weight: 10
  hidden_dim: 512
  dim_feedforward: 3200
  lr_backbone: 1e-5
  backbone: 'resnet18'
  enc_layers: 4
  dec_layers: 7
  nheads: 8
  temporal_agg: True
  state_dim: *state_dim
  action_dim: *action_dim

train_config:
  seed: 42
  num_epochs: 500000
  batch_size_val: 8
  batch_size_train: 8
  eval_ckpt_name: 'policy_epoch_90000_seed_42.ckpt'
  checkpoint_dir: CHECKPOINT_DIR
  wandb: True
"""

# Load the YAML content
config = yaml.safe_load(yaml_content)

# Print the loaded configuration to verify values are reused
print(config)

# Accessing the reused values directly to verify
print("Policy Config State Dim:", config['policy_config']['state_dim'])
print("Policy Config Action Dim:", config['policy_config']['action_dim'])