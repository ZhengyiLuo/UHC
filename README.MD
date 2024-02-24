# UHC: Universal Humanoid Controller

TODOS:

- [x] Tutorial for smpl_robot
- [x] Runnable and trainalbe code for the SMPL model for motion imitation 
- [x] Data processing code for the AMASS dataset
- [x] [Perpetual Humanoid Controller](https://github.com/ZhengyiLuo/PerpetualHumanoidControl) Support (No RFC model). -> Please refer to [PHC-MJX](https://github.com/ZhengyiLuo/PHC_MJX).




<div float="center">
  <img src="assets/gif/teaser.gif" />
</div>

## News 🚩

[Feb 24, 2024] [PHC-MJX](https://github.com/ZhengyiLuo/PHC_MJX) and [SMPLSim](https://github.com/ZhengyiLuo/SMPLSim) has been released. It is a porting of this codebase into the newest MuJoCo, and also incorporating lessons learned in PHC. 

[Oct 15th, 2023 ] [PHC](https://github.com/ZhengyiLuo/PHC) (motion tracking in Isaac gym without residual force) has been released! Please go to PHC for the newest on motion tracking!

[June 5th, 2023 ] Since there is a growing interest in using smpl_robot in Isaac (e.g. in [Trace & Pace](https://research.nvidia.com/labs/toronto-ai/trace-pace/)), I uploaded `smpl_local_robot.py` that is compatible with Isaac. 

[March 31, 2023 ] Adding the implicit-shape model. Implicit RFC + training with different SMPL body shapes. Adding AMASS data processing code.

[February 24, 2023 ] Full code runnable

[February 22, 2023 ] Evaluation code released



## Introduction 

In this project, we develop the Universal Humanoid Controller used in our projects: [embodiedpose](https://zhengyiluo.github.io/projects/embodied_pose/), [kin_poly](https://zhengyiluo.github.io/projects/kin_poly/), and [agent_design](https://zhengyiluo.github.io/projects/agent_design/). It is a physics-based humanoid controller trained with Reinforcement Learning to imitate reference human motion. UHC is task-agnostic and only takes reference frames as input. We uses the MuJoCo simulator for this repository. It relives heavily on [residual force control](https://github.com/Khrylx/RFC) for keeping the humanoid stable, and we are actively working on relying less on this dependency. Here are a few highlights of the controller:

- Supports controlling humanoids constructed from SMPL, SMPL-H, and SMPL-X models, of all genders and body shapes.
- Causal and takes in one-frame of reference frame as input.
- Supports optimizing the humanoid's body shape paramters based on [Transform2Act](https://github.com/Khrylx/Transform2Act) and [agent_design](https://zhengyiluo.github.io/projects/agent_design/).
- Can support simulating mulitple humanoids in the same scene, though only as a proof of concept. 

## Dependencies

To create the environment, follow the following instructions: 

1. Create new conda environment and install pytroch:
```
conda create -n uhc python=3.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia # install pytorch
pip install -r requirements.txt
```

2. Download and setup mujoco: [Mujoco](http://www.mujoco.org/)
```
wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz
tar -xzf mujoco210-linux-x86_64.tar.gz
mkdir ~/.mujoco
mv mujoco210 ~/.mujoco/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin
```

## SMPL Robot

The SMPL robot, an adaption of Robot class from [Transform2Act](https://github.com/Khrylx/Transform2Act), is an automatic humanoid generation class that supports SMPL, SMPL-H, and SMPL-X models. It creates XML file humanoid model for MuJoCo simulator, and can create humanoid of different gender and body shape. It supports both capsule based and mesh based models. We use SMPL Robot to create SMPL models to simulate on-the-fly when training our UHC with different body shapes. To run SMPL Robot, use:

``` 

python uhc/smpllib/smpl_robot.py

```

To generate Isaac Gym comptialbe humanoid models, use:

```
python uhc/smpllib/smpl_local_robot.py 
```

This model supports mesh and non-mesh based humanoid used in [Trace & Pace](https://research.nvidia.com/labs/toronto-ai/trace-pace/). There is a number of setting you can play with like "upright_start", which generates a humanoid that that starts in an upright position facing x-direcation (default SMPL faces upward). 

You will need to have downloaded smpl model files from [SMPL](https://smpl.is.tue.mpg.de/), [SMPL-H](https://mano.is.tue.mpg.de/download.php), and [SMPL-X](https://smpl-x.is.tue.mpg.de/downloads). Put them in the `data/smpl` folder, unzip them into 'data/smpl' folder. Please download the v1.1.0 version, which contains the neutral humanoid. Rename the files `basicmodel_neutral_lbs_10_207_0_v1.1.0`, `basicmodel_m_lbs_10_207_0_v1.1.0.pkl`, `basicmodel_f_lbs_10_207_0_v1.1.0.pkl` to `SMPL_NEUTRAL.pkl`, `SMPL_MALE.pkl` and `SMPL_FEMALE.pkl`. The file structure should look like this:

```

|-- data
    |-- smpl
        |-- SMPL_FEMALE.pkl
        |-- SMPL_NEUTRAL.pkl
        |-- SMPLH_MALE.pkl
        |-- SMPLH_FEMALE.pkl
        |-- SMPLH_NEUTRAL.pkl
        |-- SMPLX_MALE.pkl
        |-- SMPLX_FEMALE.pkl
        |-- SMPLX_NEUTRAL.pkl
        |-- SMPLX_MALE.pkl
        
```


## Data processing for training & evaluating UHC

UHC is trained on the AMASS dataset. First, download the AMASS dataset from [AMASS](https://amass.is.tue.mpg.de/). Then, run the following script on the unzipped data:
 

```
python uhc/data_process/process_amass_raw.py
```

which dumps the data into the `amass_db_smplh.pt` file. Then, run 

```
python uhc/data_process/process_amass_db.py
```

For processing your own SMPL data for evaluation, you can refer to 
```
python uhc/data_process/process_smpl_data.py
```


## Trained models

Download pretrained models: 
- Implicit RFC + Shape variation: [download](https://drive.google.com/file/d/16FI_Ga2iE3zfP93_XgRSVVCTL6zkDEJ_/view?usp=share_link)
- Implicit RFC: [download](https://drive.google.com/file/d/1CA3lKhIEOwzhX7IMdu_B_QnjrXfPXYnM/view?usp=share_link)
- Explicit RFC [download](https://drive.google.com/file/d/1wdsS0wRji3vFY6xvORptVdFEJLyY_I1T/view?usp=share_link)

You can also use the download_data script to download the models:
```
bash download_data.sh
```

## Evaluation
```
python scripts/eval_uhc.py --cfg uhc_implicit --epoch 19000 --data sample_data/amass_copycat_take5_test_small.pkl
python scripts/eval_uhc.py --cfg uhc_implicit_shape --epoch 4700 --data sample_data/amass_copycat_take5_test_small.pkl
python scripts/eval_uhc.py --cfg uhc_explicit --epoch 5000 --data sample_data/amass_copycat_take5_test_small.pkl
```

For computing statistics (mpjpe, success rate, etc.), use the `--mode stats` command. 

## Training models

```
python scripts/train_uhc.py --cfg uhc_implicit_shape
```

## Viewer Shortcuts

| Keyboard | Function |
| ---- | --- |
| Q | Next sample |
| Space | pause |
| B | hide expert |
| N | seperate expert and imitation |
| M | hide imitation |
| T | Record screenshot|
| V | Record video|

## Citation
If you find our work useful in your research, please cite our papers [embodiedpose](https://zhengyiluo.github.io/projects/embodied_pose/),  [kin_poly](https://zhengyiluo.github.io/projects/kin_poly/), and [agent_design](https://zhengyiluo.github.io/projects/agent_design/).
```
@inproceedings{Luo2022EmbodiedSH,
  title={Embodied Scene-aware Human Pose Estimation},
  author={Zhengyi Luo and Shun Iwase and Ye Yuan and Kris Kitani},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022}
}

@inproceedings{Luo2021DynamicsRegulatedKP,
  title={Dynamics-Regulated Kinematic Policy for Egocentric Pose Estimation},
  author={Zhengyi Luo and Ryo Hachiuma and Ye Yuan and Kris Kitani},
  booktitle={Advances in Neural Information Processing Systems},
  year={2021}
}

@article{Luo2022FromUH,
  title={From Universal Humanoid Control to Automatic Physically Valid Character Creation},
  author={Zhengyi Luo and Ye Yuan and Kris M. Kitani},
  journal={ArXiv},
  year={2022},
  volume={abs/2206.09286}
}
``` 


## References
This repository is built on top of the following amazing repositories:
* Part of the UHC code is from: [rfc](https://github.com/Khrylx/RFC)
* SMPL models and layer is from: [SMPL-X model](https://github.com/vchoutas/smplx)
* Feature extractors are from: [SPIN](https://github.com/nkolot/SPIN)
* NN modules are from (khrylib): [DLOW](https://github.com/Khrylx/DLow)
