import uuid
import os
import argparse
from collections import defaultdict
from collections import deque
from gym_minigrid.wrappers import * # Test importing wrappers
import math
import numpy as np
import pprint
import torch as th
from torch import optim
from buffer import Buffer
from envs import make_environment_func, DummyVecEnv
from models import get_models
from utils import masked_softmax


get_inputs = argparse.ArgumentParser()
get_inputs.add_argument("--algorithm", default="soft_q", type=str, help="learning algorithm")
get_inputs.add_argument("--algo-variant", default="double_q", type=str, help="learning algorithm variant")
get_inputs.add_argument("--lunch_batch_size", default=32, type=int, help="batch size used during lunch")
get_inputs.add_argument("--bs_learn", default=64, type=int, help="batch size used during learning")
get_inputs.add_argument("--buffer-device", default="cuda:0", type=str, help="device buffer is stored on")
get_inputs.add_argument("--buffer-max-n-episodes", default=1000, type=int, help="length of the device buffer")
get_inputs.add_argument("--net", default="mlp", type=str, help="device models are stored on")
get_inputs.add_argument("--net-device", default="cuda:0", type=str, help="device models are stored on")
get_inputs.add_argument("--env", default="simple_4way_grid", type=str, help="environment string")
get_inputs.add_argument("--env-arg-scenario-name", default="MiniGrid-Empty-5x5-v0", type=str, help="environment scenario name")
get_inputs.add_argument("--env-arg-max-steps", default=20, type=int, help="environment max steps")
get_inputs.add_argument("--env-arg-grid-dim", default=[4,4], nargs=2, type=int, help="grid dimensions tuple(x,y) of grid env")
get_inputs.add_argument("--epsilon-final", default=0.05, type=float, help="epsilon schedule final")
get_inputs.add_argument("--epsilon-timescale", default=10000, type=float, help="epsilon schedule timescale")
get_inputs.add_argument("--exp-name", required=True, default="unnamed", type=str, help="learning algorithm")
get_inputs.add_argument("--eval-every-x-episodes", default=3000, type=int, help="eval every x episodes")
get_inputs.add_argument("--gamma", default=0.99, type=float, help="gamma [0,1] (discount parameter)")
get_inputs.add_argument("--learn-every-x-episodes", default=1, type=int, help="learn every x episodes")
get_inputs.add_argument("--learn-every-x-steps", default=-1, type=int, help="learn every x episodes")
get_inputs.add_argument("--train-learning-range", default=10, type=int, help="learn x times per learn update")
get_inputs.add_argument("--log-betas-range-lunch", default=(0.01, 7.0), type=float, nargs=2,  help="range of log betas to be sampled during lunch (soft_q only)")
get_inputs.add_argument("--log-betas-eval", default=[0.01, 1, 1.5] + list(range(2, 7)), type=float, nargs="+", help="range of log betas to be sampled during lunch")
get_inputs.add_argument("--lr", default=1E-4, type=float, help="learning rate")
get_inputs.add_argument("--n-episodes-lunch-total", default=1000000, type=int, help="number of episodes to lunch in total")
get_inputs.add_argument("--n-episode-lunch-per-outer-loop", default=100, type=int, help="number of episodes to lunch in total")
get_inputs.add_argument("--n-episodes-eval", default=50, type=int, help="number of episodes to evaluate on")
get_inputs.add_argument("--n-vec-envs", default=1, type=int, help="number of vec envs to run")
get_inputs.add_argument("--update-target-net-every-x-episodes", default=10, help="update target net every x episodes")
get_inputs.add_argument("--input-t", default=True, type=bool, help="update target net every x episodes")


if __name__ == '__main__':
    args = get_inputs.parse_args()
    pprint.pprint(vars(args))
    make_env_fn = make_environment_func(args.env, args) # get make env fn
    # Create other_shapes_dct
    other_shapes_dct = {}
    if args.algorithm == "soft_q":
        other_shapes_dct["log_beta"] = (1,)
    if args.input_t:
        other_shapes_dct["t"] = (1,)
    reward_max = -10000
    # create vec envreward_max
    vec_env = DummyVecEnv(env_fn=make_env_fn, batch_size=args.lunch_batch_size)
    eval_vec_env = DummyVecEnv(env_fn=make_env_fn, batch_size=1)

    exphash = uuid.uuid4().hex[:6]

    # create models
    models, models_get_params = get_models(net_tag=args.net,
                                           n_actions=vec_env.get_action_space()["player1"].n,
                                           max_steps=args.env_arg_max_steps,
                                           obs_shape=vec_env.reset()["player1"][0].shape,
                                           other_shapes_dct=other_shapes_dct,
                                           device=args.net_device)

    # Create network optimizer : Adam
    optimizer = optim.Adam(models_get_params(), lr=args.lr)

    # Set up epsilon fn
    def epsilon_fn(ep_t):
        decay_rate = -ep_t / args.epsilon_timescale
        epsilon = max(1.0 * math.exp(decay_rate), args.epsilon_final)
        return epsilon

    def manipulate_tensor(last_dct, key, device):
        vals = [v[key][-1] for v in last_dct]
        tens = th.stack(vals).to(device)
        return tens

    def manipulate_stack(dct, device):
        new_dct = {}
        for k, v in dct.items():
            if isinstance(v, (list, tuple)):
                new_dct[k] = th.stack(v).to(device)
            else:
                new_dct[k] = v
        return new_dct


    buffer_scheme = {"obs": vec_env.get_obs_space()["player1"].high.shape,
                     "action": (vec_env.get_action_space()["player1"].n,),
                     "avail_actions": (vec_env.get_action_space()["player1"].n,),
                     "reward": (1,),
                     "epsilon": (1,),
                     "log_beta": (1,),
                     }
    if args.input_t:
        buffer_scheme["t"] = (1,)
    buffer = Buffer(buffer_scheme, buffer_len=args.buffer_max_n_episodes, buffer_device=args.buffer_device)

    lnch_dct_ls = [{"done": -1} for _ in range(args.lunch_batch_size)]
    episode_number = 0
    train_mex_eps = 0    
    train_mex_step = 0
    train_print_sp = 0
    test_max_eps = 0
    test_max_step = 0
    stp = 0

    episodes_plot = []
    steps_plot = []
    mean_return_reward_train_plot = []
    loss_plot = []
    test_reward_plot = np.empty((0,len(args.log_betas_eval)))

    lunch_stats = {"return": deque(maxlen = 100), "epsilon": deque(maxlen=100)}
    while episode_number < args.n_episodes_lunch_total:

################################ LUNCH START ################################################################
        for s in range(args.n_episode_lunch_per_outer_loop):
            # Reset all
            for indx in range(args.lunch_batch_size):
                if lnch_dct_ls[indx]["done"] in [1, -1]:
                    if lnch_dct_ls[indx]["done"] == 1:
                        manipulated_stack = manipulate_stack(lnch_dct_ls[indx], device=args.net_device)
                        buffer.insert(manipulated_stack)
                        reward_sum = sum(lnch_dct_ls[indx]["reward"])
                        lunch_stats["return"].append(th.zeros(1, device=args.net_device) + reward_sum)
                        lunch_stats["epsilon"].append(th.mean(th.stack(lnch_dct_ls[indx]["epsilon"])))
                        episode_number += 1
                    log_beta = (args.log_betas_range_lunch[0] - args.log_betas_range_lunch[1]) * \
                            th.rand((1,), device=args.net_device) + args.log_betas_range_lunch[1]
                    RST = vec_env.reset(indx)["player1"]
                    avail_ac = vec_env.get_available_actions(idx=indx)["player1"].to(args.net_device)
                    lnch_dct_ls[indx] = {
                        "done": 0,
                        "obs": [RST],
                        "action": [],
                        "reward": [],
                        "log_beta": [th.zeros(1, device=args.net_device) + log_beta],
                        "avail_actions": [avail_ac],
                        "epsilon": []
                    }
                    if args.input_t:
                        lnch_dct_ls[indx]["t"] = [th.zeros(1, device=args.net_device)]


            # sample greedy actions
            other_dct = {"log_beta": manipulate_tensor(lnch_dct_ls, "log_beta", args.net_device)}
            if args.input_t:
                other_dct["t"] = manipulate_tensor(lnch_dct_ls, "t", args.net_device)
            q_values = models["net"](manipulate_tensor(lnch_dct_ls, "obs", args.net_device), other_dct)["out"]
            ro_avail_actions = manipulate_tensor(lnch_dct_ls, "avail_actions", args.net_device)

            policy = masked_softmax(q_values,
                                    mask=(ro_avail_actions == 0),
                                    beta=th.exp(manipulate_tensor(lnch_dct_ls, "log_beta", args.net_device)),
                                    dim=-1)
            ro_action = th.multinomial(policy, 1)

            # add action exploration
            ro_epsilon = epsilon_fn(episode_number)
            ro_epsilon0 = ro_action.clone().float().uniform_()
            ro_action[ro_epsilon0 < ro_epsilon] = th.multinomial(ro_avail_actions.float(), 1)[ro_epsilon0 < ro_epsilon]
            obs, reward, done, info = vec_env.step(ro_action)
            stp += args.lunch_batch_size

            # store step results
            for indx in range(args.lunch_batch_size):
                lnch_dct_ls[indx]["action"].append(ro_action[indx])
                lnch_dct_ls[indx]["obs"].append(obs["player1"][indx])
                lnch_dct_ls[indx]["reward"].append(reward[indx])
                lnch_dct_ls[indx]["done"]  = done[indx]
                lnch_dct_ls[indx]["avail_actions"].append(vec_env.get_available_actions(idx=indx)["player1"].to(args.net_device))
                lnch_dct_ls[indx]["epsilon"].append(th.zeros(1, device=args.net_device) + ro_epsilon)
                lnch_dct_ls[indx]["log_beta"].append(th.zeros(1, device=args.net_device) + log_beta)
                if args.input_t:
                    lnch_dct_ls[indx]["t"].append(lnch_dct_ls[indx]["t"][-1] + 1)
            pass

        if len(lunch_stats["return"]) == 100:
            episodes_plot.append(episode_number)
            steps_plot.append(stp)
            mean_return_reward_train_plot.append(th.mean(th.stack(list(lunch_stats["return"]))).item())  
            print("{} episodes, {} steps: mean train reward= {:.2f} @ epsilon(average): {:.2f}".format(
                episode_number,
                stp,
                th.mean(th.stack(list(lunch_stats["return"]))).item(),
                th.mean(th.stack(list(lunch_stats["epsilon"]))).item()))
        ################################ ROLLOUTS STOP #################################################################


        ################################ TRAINING START ################################################################
        if ( (args.learn_every_x_steps == -1 and episode_number - train_mex_eps >= args.learn_every_x_episodes) or \
             (args.learn_every_x_steps != -1 and stp - train_mex_step >= args.learn_every_x_steps) ) and \
                (buffer.size(mode="transitions") >= args.bs_learn):

            train_mex_eps = episode_number
            train_mex_step = stp

            for _ in range(args.train_learning_range):
                samples = {k: v.to(args.net_device) for k, v in buffer.sample(args.bs_learn, mode="transitions").items()}
                other_dct = {}
                other_dict_next = {}
                
                if args.input_t:
                    other_dct["t"] = samples["t"]
                    other_dict_next["t"] = samples["t"] + 1
                    
                if args.algorithm == "soft_q":
                    other_dct["log_beta"] = samples["log_beta"]
                    other_dict_next["log_beta"] = samples["log_beta"]
                
                q_values = models["net"](samples["obs"], other_dct, is_sequence=False)["out"]
                q_value_taken = q_values.gather(-1, samples["action"].long())
                q_value_taken = q_value_taken
                
                target_q_values = models["target_net"](samples["next_obs"], other_dict_next, is_sequence=False)["out"]
                target_q_values[~samples["next_avail_actions"]] = -1E20
                target_q_value = (1. / th.exp(samples["log_beta"])) * th.logsumexp(
                    th.exp(samples["log_beta"]) * target_q_values,
                    dim=-1, keepdim=True)


                # For a terminal state the target_value is always 0
                terminal_state_mask = samples["next_obs_is_terminal"].bool()
                target_q_value[terminal_state_mask] = 0.0

                target_value = args.gamma * target_q_value + samples["reward"].unsqueeze(-1) # reward is always in previous timestep

                loss = (target_value.detach() - q_value_taken) ** 2
                loss = loss.mean()

                if stp - train_print_sp > 1000:
                    loss_plot.append(loss.mean())   
                    print("Loss (t_ep: {} steps: {}): {}".format(episode_number, stp, loss.mean()))
                    train_print_sp = stp

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Update target network
        if episode_number - test_max_step >= args.update_target_net_every_x_episodes:
            test_max_step = episode_number
            print("training model updated.")
            models["target_net"].load_state_dict(models["net"].state_dict())
#### Train Done ----> Start test
        if episode_number - test_max_eps >= args.eval_every_x_episodes:
            print("new model saved.")
            path = os.path.join("results", "{}__{}".format(args.exp_name, exphash))
            if not os.path.exists(path):
                os.makedirs(path)

            print("test start:")
            test_max_eps = episode_number
            for log_beta in (args.log_betas_eval if args.algorithm == "soft_q" else [0]):
                all_rewards = []
                all_ep_lengths = []
                traj_dict = defaultdict(lambda:0)
                for s in range(args.n_episodes_eval):
                    current_rewards = []
                    eval_vec_env = DummyVecEnv(env_fn=make_env_fn, batch_size=1) # NEW!!!
                    eval_obs = eval_vec_env.reset()["player1"].unsqueeze(0)
                    other_dct = {}
                    if args.algorithm == "soft_q":
                        other_dct["log_beta"] = th.FloatTensor([[log_beta]]).to(args.net_device)
                    eval_t = 0
                    act_lst = []
                    max_eval_steps = 3000
                    while True:
                        if args.input_t:
                            other_dct["t"] = th.FloatTensor([[eval_t]])
                        q_values = models["net"](eval_obs.to(args.net_device), other_dct)["out"]
                        avail_actions = eval_vec_env.get_available_actions()["player1"].to(args.net_device)

                        if args.algorithm == "soft_q":
                            policy = masked_softmax(q_values,
                                                    mask=(avail_actions == 0),
                                                    beta=th.exp(other_dct["log_beta"]),
                                                    dim=-1)
                            next_action = th.multinomial(policy.squeeze(0), 1)
                        else:
                            q_values[~avail_actions] = -1E20
                            next_action = th.argmax(q_values, -1, keepdim=True).detach()

                        act_lst.append(next_action[0].item())
                        eval_obs, reward, done, info = eval_vec_env.step(next_action)
                        eval_obs = eval_obs["player1"]
                        current_rewards.append(reward.item())
                        eval_t += 1
                        if False:
                            eval_vec_env.render()
                        if done[0] or eval_t > max_eval_steps:
                            all_rewards.append(np.sum(current_rewards))
                            all_ep_lengths.append(eval_t)
                            traj_dict[tuple(act_lst)] += 1
                            break

                if args.algorithm == "soft_q":
                    test_reward_plot = np.append(test_reward_plot, np.mean(all_rewards))
                    print("Mean episode reward test (log beta: {:.2f}): {:.2f} len: {:.2f}  number of trajectories: {:.2f}".format(log_beta,
                                                                                                                       np.mean(all_rewards),
                                                                                                                       np.mean(all_ep_lengths),
                                                                                                                       len(traj_dict.values())))

                else:
                    print("Mean episode reward test: {:.2f} len: {:.2f}".format(np.mean(all_rewards), np.mean(all_ep_lengths)))

            test_reward_plot = test_reward_plot.reshape((-1,len(args.log_betas_eval)))
            
            mean_reward_all = np.mean(all_rewards)
            if mean_reward_all >= reward_max:
                reward_max = mean_reward_all
                print("Saving model...")
        ################################ EVAL STOP #####################################################################
    np.savetxt('./CSV/return_train.txt' , mean_return_reward_train_plot)

    cpu_tensor_list = [tensor.detach().cpu() for tensor in loss_plot]
    numpy_array_list = [tensor.numpy() for tensor in cpu_tensor_list]
    np.savetxt('./CSV/loss_train.txt' , numpy_array_list)
    np.savetxt('./CSV/test_reward.txt' , test_reward_plot)




