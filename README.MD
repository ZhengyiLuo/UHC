# UHC: Universal Humanoid Controller

> Repo under construction. ## Introduction

## Introduction 

In this project, we develop the Universal Humanoid Controller used in our projects: [embodiedpose](https://zhengyiluo.github.io/projects/embodied_pose/), [kin_poly](https://zhengyiluo.github.io/projects/kin_poly/), and [agent_design](https://zhengyiluo.github.io/projects/agent_design/). 

## Dependencies

To create the environment, follow the following instructions: 

1. Create new conda environment and install pytroch:
```
conda create -n uhc python=3.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
```

2. Download and setup mujoco: [Mujoco](http://www.mujoco.org/)
```
wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz
tar -xzf mujoco210-linux-x86_64.tar.gz
mkdir ~/.mujoco
mv mujoco210 ~/.mujoco/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin
```




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

