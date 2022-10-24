from uhc.khrylib.rl.core import TrajBatch
import numpy as np


class TrajBatchEgo(TrajBatch):
    def __init__(self, memory_list):
        super().__init__(memory_list)
        self.v_metas = np.stack(next(self.batch))
        self.gt_target_qpos = np.stack(next(self.batch))
        self.curr_qpos = np.stack(next(self.batch))
        self.res_qpos = np.stack(next(self.batch))
