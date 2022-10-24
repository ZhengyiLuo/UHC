from .kin_net import KinNet
from .kin_net_goal import KinNetGoal
from .super_net import SuperNet

model_dict = {
    "kin_net": KinNet,
    "kin_net_goal": KinNetGoal,
    "super_net": SuperNet,
}

from .kin_policy import KinPolicy
from .kin_policy_goal import KinPolicyGoal

policy_dict = {"kin_policy": KinPolicy, "kin_policy_goal": KinPolicyGoal}
