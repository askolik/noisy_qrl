import gym
import tensorflow as tf

from utils.layers import ReUploadingPQC, Alternating
from utils.storage import ScoresCheckpoint, MetaCallback

tf.get_logger().setLevel('ERROR')

import cirq
import numpy as np
import tqdm
from functools import reduce
import collections
from collections import defaultdict
import statistics
import time

# Set seed for experiment reproducibility
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)


def generate_model_policy(
        qubits, n_layers, n_actions, beta, observables,
        noise_p=0., param_perturbation=0., n_shots=0):
    input_tensor = tf.keras.Input(shape=(len(qubits),), dtype=tf.dtypes.float32, name='input')

    re_uploading_pqc = ReUploadingPQC(
        qubits, n_layers, observables, noise_p=noise_p,
        param_perturbation=param_perturbation, n_shots=n_shots)([input_tensor])

    process = tf.keras.Sequential([
        Alternating(n_actions),
        tf.keras.layers.Lambda(lambda x: x * beta),
        tf.keras.layers.Softmax()
    ], name="observables-policy")
    policy = process(re_uploading_pqc)
    model = tf.keras.Model(inputs=[input_tensor], outputs=policy, name="QuantumActor")

    return model


def compute_returns(rewards_history, gamma):
    """Compute discounted returns with discount factor `gamma`."""
    returns = []
    discounted_sum = 0
    for r in rewards_history[::-1]:
        discounted_sum = r + gamma * discounted_sum
        returns.insert(0, discounted_sum)

    # Normalize them for faster and more stable learning
    returns = np.array(returns)
    returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)
    returns = returns.tolist()

    return returns


def logger(env_name, noise_p=0., n_qubits=4, n_layers=5, beta=1.0,
           gamma=0.99, batch_size=10, max_episodes=1500, param_perturbation = 0.):
    print("\n===================================================")
    print("SIMULATION DETAILS ")
    print("===================================================")
    print(f"Env = {env_name}")
    print(f"Noise probability = {noise_p}")
    print(f"N. qubits = {n_qubits}")
    print(f"N. layers = {n_layers}")
    print(f"Beta = {beta}")
    print(f"Discount factor (gamma) = {gamma}")
    print(f"Batch size = {batch_size}")
    print(f"Max episodes until termination = {max_episodes}")
    print(f"Param perturbation (gaussian) = {param_perturbation}")
    print("===================================================\n")


def gather_episodes(state_bounds, n_actions, model, n_episodes, env_name):
    trajectories = [defaultdict(list) for _ in range(n_episodes)]
    envs = [gym.make(env_name) for _ in range(n_episodes)]

    done = [False for _ in range(n_episodes)]
    states = [e.reset() for e in envs]

    while not all(done):
        unfinished_ids = [i for i in range(n_episodes) if not done[i]]
        normalized_states = [s / state_bounds for i, s in enumerate(states) if not done[i]]

        for i, state in zip(unfinished_ids, normalized_states):
            trajectories[i]['states'].append(state)

        states = tf.convert_to_tensor(normalized_states)
        action_probs = model([states])

        states = [None for i in range(n_episodes)]
        for i, policy in zip(unfinished_ids, action_probs.numpy()):
            action = np.random.choice(n_actions, p=policy)
            states[i], reward, done[i], _ = envs[i].step(action)
            trajectories[i]['actions'].append(action)
            trajectories[i]['rewards'].append(reward)

    return trajectories


# @tf.function  # decorator should make code faster but causes issues on cluster
def reinforce_update(states, actions, returns, model, batch_size, optimizers, w_idx):
    states = tf.convert_to_tensor(states)
    actions = tf.convert_to_tensor(actions)
    returns = tf.convert_to_tensor(returns)

    with tf.GradientTape() as tape:
        tape.watch(model.trainable_variables)
        logits = model(states)
        p_actions = tf.gather_nd(logits, actions)
        log_probs = tf.math.log(p_actions)
        loss = tf.math.reduce_sum(-log_probs * returns) / batch_size
    grads = tape.gradient(loss, model.trainable_variables)
    for optimizer, w in zip(optimizers, w_idx):
        optimizer.apply_gradients([(grads[w], model.trainable_variables[w])])


def train_cp_pg(noise_p=0., noise_type='depolarize', n_qubits=4, n_layers=5, beta=1.,
                gamma=0.99, batch_size=10, max_episodes=1500,
                lr_in=0.1, lr_var=0.01, lr_out=0.1, param_perturbation=0.,
                checkpoint=False, checkpoint_path='data/', save_as='test'):

    if checkpoint:
        meta = {
            'noise_p': noise_p,
            'noise_type': noise_type,
            'n_layers': n_layers,
            'beta': beta,
            'gamma': gamma,
            'batch_size': batch_size,
            'learning_rate_in': lr_in,
            'learning_rate_var': lr_var,
            'learning_rate_out': lr_out,
            'param_perturbation': param_perturbation,
        }

        scores_checkpoint = ScoresCheckpoint(checkpoint_path, save_as)
        meta_callback = MetaCallback(meta, checkpoint_path, save_as)

    env_name = "CartPole-v0"
    env = gym.make(env_name)
    env.seed(seed)
    n_actions = env.action_space.n
    state_bounds = np.array([2.4, 2.5, 0.21, 2.5])  # Max allowed values for the state

    logger(env_name, noise_p=noise_p, n_qubits=n_qubits, n_layers=n_layers,
           beta=beta, gamma=gamma, batch_size=batch_size, max_episodes=max_episodes,
           param_perturbation=param_perturbation)

    qubits = cirq.GridQubit.rect(1, n_qubits)
    ops = [cirq.Z(q) for q in qubits]
    observables = [reduce((lambda x, y: x * y), ops)]

    model = generate_model_policy(
        qubits, n_layers, n_actions, 1.0, observables,
        noise_p=noise_p, param_perturbation=param_perturbation)

    model.summary()

    optimizer_in = tf.keras.optimizers.Adam(learning_rate=lr_in, amsgrad=True)
    optimizer_var = tf.keras.optimizers.Adam(learning_rate=lr_var, amsgrad=True)
    optimizer_out = tf.keras.optimizers.Adam(learning_rate=lr_out, amsgrad=True)
    optimizers = [optimizer_in, optimizer_var, optimizer_out]

    # Assign the model parameters to each optimizer
    w_in, w_var, w_out = 1, 0, 2
    w_idx = [w_in, w_var, w_out]

    ###########################################################################
    # TRAINING
    reward_threshold = 195  # Target threshold to consider env solved
    min_episodes_criterion = 100  # Evaluate average reward over min_episodes_criterion episodes (100 to consider env solved)
    episodes_reward: collections.deque = collections.deque(maxlen=min_episodes_criterion)
    episode_history = []

    start_time = time.time()
    with tqdm.trange(max_episodes // batch_size) as t:
        for batch in t:
            episodes = gather_episodes(state_bounds, n_actions, model, batch_size, env_name)
            states = np.concatenate([ep['states'] for ep in episodes])
            actions = np.concatenate([ep['actions'] for ep in episodes])
            rewards = [ep['rewards'] for ep in episodes]
            returns = np.concatenate([compute_returns(ep_rwds, gamma) for ep_rwds in rewards])
            returns = np.array(returns, dtype=np.float32)

            id_action_pairs = np.array([[i, a] for i, a in enumerate(actions)])

            reinforce_update(
                states, id_action_pairs, returns, model, batch_size, optimizers, w_idx)

            for ep_rwds in rewards:
                tot_rew = np.sum(ep_rwds)
                episode_history.append(tot_rew)
                episodes_reward.append(tot_rew)

            running_reward = statistics.mean(episodes_reward)

            t.set_description(f'Episode {(batch + 1) * batch_size}')
            t.set_postfix(
                episode_reward=tot_rew, running_reward=running_reward)

            if checkpoint:
                model.save_weights(checkpoint_path + save_as)
                scores_checkpoint.on_epoch_end(t, {'scores': episode_history})
                meta_callback.on_epoch_end(t)

            if running_reward >= reward_threshold:
                print("Env solved!")

                if checkpoint:
                    meta['env_solved'] = True
                    meta_callback.on_epoch_end(t, meta)

                break

    print(f"\nExec time {(time.time() - start_time)} seconds")
    print(f'Finished at episode {(batch + 1) * batch_size}: average reward: {running_reward:.2f}!')

    return episode_history, {(batch + 1) * batch_size}, running_reward
