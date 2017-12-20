
Set OPENPOSE_MODEL env var

symlink qwop_env.py to your gym/envs folder

ln -s qwop/infer_img.py tf-openpose/infer_img.py

modify classic_control/__init__.py to have

from gym.envs.classic_control.qwop_env import QWOPEnv

add to registration.py

register(
    id='QWOP-v0',
    entry_point='gym.envs.classic_control:QWOPEnv',
    max_episode_steps=100000000,
    reward_threshold=100000000000,
)