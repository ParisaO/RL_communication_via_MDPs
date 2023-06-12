from collections import defaultdict
import torch as th

import multiprocessing as mp
from vec_env import CloudpickleWrapper, clear_mpi_env_vars


class DummyVecEnv():

    def __init__(self, env_fn, batch_size):
        self.batch_size = batch_size
        self.envs = [env_fn.x() if isinstance(env_fn, CloudpickleWrapper) else env_fn() for _ in range(batch_size)]
        pass

    def step(self, actions, a=None):
        if a is None:
            obs_lst = []
            reward_lst = []
            done_lst = []
            info_lst = []
            for i in range(self.batch_size):
                obs, reward, done, info = self.envs[i].step(actions[i])
                obs_lst.append(obs)
                reward_lst.append(reward)
                done_lst.append(done)
                info_lst.append(info[0])

            return th.stack(obs_lst) if not isinstance(obs_lst[0], dict) else {k:th.stack([o[k] for o in obs_lst]) for k,v in obs_lst[0].items()}, \
                   th.FloatTensor(reward_lst), \
                   done_lst, \
                   info_lst
        else:
            obs, reward, done, info = self.envs[a].step(actions)
            return obs, th.FloatTensor([reward]), done, info

    def reset(self, idx=None):
        if idx is None:
            idx_lst = list(range(self.batch_size))
        else:
            idx_lst = [idx] if isinstance(idx, int) else idx
        obs_lst = []
        for i in idx_lst:
            obs = self.envs[i].reset()
            obs_lst.append(obs)
        if len(idx_lst) == 1:
            return obs_lst[0] if not isinstance(obs_lst[0], dict) else {k: v for  k, v in obs_lst[0].items()}
        else:
            return th.stack(obs_lst) if not isinstance(obs_lst[0], dict) else {k:th.stack([o[k] for o in obs_lst]) for k,v in obs_lst[0].items()}

    def render(self, **kwargs):
        img = self.envs[0].render('rgb_array')
        return img

    def close(self):
        for i in range(self.batch_size):
            self.envs[i].close()

    def get_obs_space(self):
        return self.envs[0].get_obs_space()

    def get_action_space(self):
        return self.envs[0].get_action_space()

    def get_available_actions(self, idx=None):
        if idx is None:
            idx_lst = list(range(self.batch_size))
        else:
            idx_lst = [idx] if isinstance(idx, int) else idx

        avail_actions_dct = defaultdict(lambda: [])
        for i in idx_lst:
            avail_actions = self.envs[i].get_available_actions()
            for k, v in avail_actions.items():
                avail_actions_dct[k].append(v)
        if len(idx_lst) == 1:
            return {k:v[0] for k, v in avail_actions_dct.items()}
        else:
            return {k:th.stack(v) for k, v in avail_actions_dct.items()}

    def get_trajectory(self, idx=None):
        if idx is None:
            idx_lst = list(range(self.batch_size))
        else:
            idx_lst = [idx] if isinstance(idx, int) else idx
        return [self.envs[_id].trajectory for _id in idx_lst]
    pass


def dummy_vec_env_worker(remote, parent_remote, env_fn_wrappers, batch_size=None):
    vec_env = DummyVecEnv(env_fn=env_fn_wrappers, batch_size=batch_size)
    parent_remote.close()
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                remote.send(vec_env.step(data))
            elif cmd == 'reset':
                remote.send(vec_env.reset(data))
            elif cmd == 'render':
                remote.send(vec_env.render())
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'get_spaces_spec':
                remote.send(CloudpickleWrapper((vec_env.envs[0].get_obs_space(),
                                                vec_env.envs[0].get_action_space())))
            elif cmd == 'get_available_actions':
                remote.send(vec_env.get_available_actions(data))
            elif cmd == 'get_trajectory':
                remote.send(vec_env.get_trajectory(data))
            else:
                raise NotImplementedError
    except KeyboardInterrupt:
        print('SubprocVecEnv worker: got KeyboardInterrupt')
    finally:
        vec_env.close()


def make_environment_func(env_tag, env_args):

    # Create environment
    env_registry = {"simple_4way_grid": Simple4WayGrid, "mini_grid": None}
    assert env_tag in env_registry, "Environment {} not in registry!".format(env_tag)

    make_env_fn = None
    if env_tag in ["simple_grid", "simple_4way_grid"]:
        def _make_env_fn():
            env = env_registry[env_tag](max_steps=env_args.env_arg_max_steps, grid_dim=env_args.env_arg_grid_dim)
            env.max_steps = env_args.env_arg_max_steps  # min(env.max_steps, max_steps)
            return env

    else:
        print("Unknown model")

    return _make_env_fn