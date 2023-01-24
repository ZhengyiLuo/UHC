from .kin_net import KinNet
from .super_net import SuperNet

model_dict = {
    "kin_net": KinNet,
    "super_net": SuperNet,
}

from .kin_policy import KinPolicy

policy_dict = {"kin_policy": KinPolicy}
