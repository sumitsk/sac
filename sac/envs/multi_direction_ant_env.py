import numpy as np
from rllab.envs.mujoco.ant_env import AntEnv
from rllab.misc.overrides import overrides
from rllab.envs.base import Step

class MultiDirectionAntEnv(AntEnv):
    @overrides
    def step(self, action):
        self.forward_dynamics(action)

        xy_velocities = self.get_body_comvel("torso")[:2]
        # rewards for speed on xy-plane (no matter which direction)
        xy_velocity = np.sqrt(np.sum(xy_velocities**2))

        lb, ub = self.action_bounds
        scaling = (ub - lb) * 0.5
        ctrl_cost = 0.5 * 1e-2 * np.sum(np.square(action / scaling))
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.model.data.cfrc_ext, -1, 1))),
        survive_reward = 0.05
        reward = xy_velocity - ctrl_cost - contact_cost + survive_reward
        state = self._state
        notdone = np.isfinite(state).all() \
            and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self.get_current_obs()
        return Step(ob, float(reward), done)