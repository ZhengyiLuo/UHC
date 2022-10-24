from .agent_copycat import AgentCopycat
from .agent_copycat_sl import AgentCopycatSL
from .agent_uhm import AgentUHM
from .agent_uhm_goal import AgentUHMGoal

agent_dict = {
    'agent_copycat': AgentCopycat,
    'agent_uhm': AgentUHM,
    'agent_uhm_goal': AgentUHMGoal, 
    "agent_copycat_sl": AgentCopycatSL
}