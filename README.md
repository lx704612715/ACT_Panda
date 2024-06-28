# Documentation for training ACT with a Panda arm

### Environment Setup
```bash
# For ACT
git clone git@github.com:lx704612715/ACT_Panda.git
cd ACT_Panda
conda create --name act python=3.9
conda activate act
pip install -r requirements.txt

git clone git@github.com:lx704612715/detr.git
cd detr
pip install -e .
cd ..
# For Diffusion
git clone --branch r2d2 git@github.com:ARISE-Initiative/robomimic.git
cd robomimic
pip install -e .
```

### Record demonstration
```bash
ssh -X robotics@controller
roslaunch panda_hybrid_automaton_manager panda_ha.launch
python3 act_panda/demonstration/record_episodes.py --task latch
```

### Align all demonstration data to the same episode length
```bash
python3 act_panda/demonstration/align_episode.py 
```

### Train ACT policy
```bash
python3 act_panda/training/train_latch.py --task latch
```

### Inference with Panda Robot
```bash
python3 act_panda/evaluation/act_controller.py
```

# Imitation Learning for 250$ robot arm
This repository contains a re-adapatation of [Action Chunking Transformer](https://github.com/tonyzhaozh/act/tree/main) that works for this [low-cost robot](https://github.com/AlexanderKoch-Koch/low_cost_robot) design (250$). 

We are sharing the repo so anyone (non-experts included) can train a robot policy after a few teleoperated demonstraions.

The sorting task in the video was trained with less than 30 demonstrations on an RTX 3080 and took less than 30min.

https://github.com/Shaka-Labs/ACT/assets/45405956/83c05915-7442-49a4-905a-273fe35e84ee

## AI training
### Setup
Create conda environment
~~~
conda create --name act python=3.9
conda activate act
~~~

Install torch (for reference we add the versions we are using)
~~~
conda install pytorch==1.13.1 torchvision==0.14.1
~~~

You can now install the requirements:
~~~
pip install -r requirements.txt
~~~

Go to `TASK_CONFIG` in `config/config.py` and change the paths of the ports that connect leader and follower robots to your computer. 

You will also need to connect a camera to your computer and point it towards the robot while collecting the data via teleoperation. You can change the camera port in the config (set to 0 by default). It's important the camera doesn't move otherwise evaluation of the policy is likely to fail. 

### Data collection
In order to collect data simply run:
~~~
python record_episodes.py --task sort
~~~
You can define the name of the task you are doing and the episodes will be stored at `data/<task>`. You can also select how many episodes to collect when running the script by passing the argument `--num_episodes 1` (set to 1 by default). After getting a hold of it you can easily do 20 tasks in a row.

Turn on the volume of your pc-- data for each episode will be recorded after you hear "Go" and it will stop when you hear "Stop".

### Train policy
We slightly re-adapt [Action Chunking Tranfosrmer](https://github.com/tonyzhaozh/act/tree/main) to account for our setup. To start training simply run:
~~~
python train.py --task sort
~~~
The policy will be saved in `checkpoints/<task>`.

### Evaluate policy
Make sure to keep the same setup while you were collecting the data. To evaluate the policy simply run:
~~~
python evaluate.py --task sort
~~~
