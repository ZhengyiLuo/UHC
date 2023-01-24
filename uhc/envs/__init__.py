from .humanoid_im import HumanoidEnv
from .humanoid_kin_v1 import HumanoidKinEnv

env_dict = {
    'humanoid_im': HumanoidEnv,
    'humanoid_kin_v1': HumanoidKinEnv
}