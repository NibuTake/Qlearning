import gym
from gym import envs
import numpy as np

def get_status(_observation):
    env_low = env.observation_space.low # 位置と速度の最小値
    env_high = env.observation_space.high #　位置と速度の最大値
    env_dx = (env_high - env_low) /100 # 40等分
    # 0〜39の離散値に変換する
    position = int((_observation[0] - env_low[0])/env_dx[0])
    velocity = int((_observation[1] - env_low[1])/env_dx[1])
    return position, velocity

def update_q_table(_q_table, _action,  _observation, _next_observation, _reward, _episode,
                    _iteration, _epoch):

    alpha = 0.3 + 0.3 * 0.98 ** (np.sqrt(_epoch))
    gamma = 0.99

    next_position, next_velocity = get_status(_next_observation)
    next_max_q_value = max(_q_table[next_position][next_velocity])

    # 行動前の状態の行動価値 Q(s,a)
    position, velocity = get_status(_observation)
    q_value = _q_table[position][velocity][_action]

    # 行動価値関数の更新
    _q_table[position][velocity][_action] = q_value + alpha * (_reward + gamma * next_max_q_value - q_value)

    return _q_table

def get_action(_env, _q_table, _observation, _episode, _iteration, _epoch):
    epsilon = 0.0025
    if np.random.uniform(0, 1) > epsilon:
        position, velocity = get_status(observation)
        _action = np.argmax(_q_table[position][velocity])
    else:
        _action = np.random.choice([0, 1, 2])
    return _action

def get_action0(_env, _q_table, _observation, _episode):
    position, velocity = get_status(observation)
    _action = np.argmax(_q_table[position][velocity])
    return _action

env = gym.make('MountainCar-v0')
observation = env.reset()

q_table = np.zeros((100, 100, 3))
rewards = []
best_qtable_list = []

last_reward = 200

for l, episode in enumerate(range(5000)):

    total_reward = 0
    observation = env.reset()

    for k, _ in enumerate(range(200)):

        # ε-グリーディ法で行動を選択
        action = get_action(env, q_table, observation, episode, k, l)

        # 車を動かし、観測結果・報酬・ゲーム終了FLG・詳細情報を取得
        next_observation, reward, done, _ = env.step(action)
        if (last_reward>199) & (k < 10):
            reward = reward + observation[1]*1

        if (done==True) & (k+1<200) :
            reward = 10

        # Qテーブルの更新
        q_table = update_q_table(q_table, action, observation, next_observation,
                                 reward, episode, k, l)
        # total_reward +=  reward * (0.98 ** k)
        total_reward +=  reward

        observation = next_observation

        if done:
            last_reward = k+1
            # doneがTrueになったら１エピソード終了
            if episode%100 == 0:
                print('episode: {}, total_reward: {}'.format(episode, total_reward))
                print('clear iteration: {}'.format(k+1))
                if k+1 < 120:
                    best_qtable_list.append([q_table])
            rewards.append(total_reward)
            break
print("Start!!")

"""
# Jupyterでmatplotlibを使用する宣言と、使用するライブラリの定義
import gym
import numpy as np
import matplotlib.pyplot as plt

# 動画の描画関数の宣言
# 参考URL http://nbviewer.jupyter.org/github/patrickmineault
# /xcorr-notebooks/blob/master/Render%20OpenAI%20gym%20as%20GIF.ipynb
from JSAnimation.IPython_display import display_animation
from matplotlib import animation
from IPython.display import display


def display_frames_as_gif(frames):
    plt.figure(figsize=(frames[0].shape[1]/72.0, frames[0].shape[0]/72.0),
               dpi=72)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames),
                                   interval=50)

    anim.save('movie_cartpole.mp4')  # 動画のファイル名と保存です
    display(display_animation(anim, default_mode='loop'))
"""


observation = env.reset()
frames = []

for i_episode in range(3):
    observation = env.reset()
    for t in range(200):
        env.render()
        frames.append(env.render(mode='rgb_array'))
        print(observation)
        action = get_action0(env, q_table, observation, i_episode)
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
