from pacman import GameState
import numpy as np
import tensorflow as tf
from baselines.a2c.utils import ortho_init, conv
from baselines.common.models import register

FRAMES = ['food', 'wall', 'capsule', 'pacman', 'ghost']


def state_to_obs_tensor(state: GameState):
    obs_tensor = np.zeros(shape=(state.data.layout.width, state.data.layout.height, len(FRAMES)))
    # food and wall
    obs_tensor[:, :, 0] = np.array(state.data.layout.food.data).astype(np.float)
    obs_tensor[:, :, 1] = np.array(state.data.layout.walls.data).astype(np.float)

    # capsule
    cap_idx1, cap_idx2 = zip(*state.data.layout.capsules)
    cap_idx3 = tuple(2 for _ in range(len(cap_idx1)))
    obs_tensor[cap_idx1, cap_idx2, cap_idx3] = 1.0

    # pacman
    pacman = [idx for is_ghost, idx in state.data.layout.agentPositions if not is_ghost]
    p_idx1, p_idx2 = zip(*pacman)
    p_idx3 = tuple(3 for _ in range(len(p_idx1)))
    obs_tensor[p_idx1, p_idx2, p_idx3] = 1.0

    # ghost
    ghost = [idx for is_ghost, idx in state.data.layout.agentPositions if is_ghost]
    g_idx1, g_idx2 = zip(*ghost)
    g_idx3 = tuple(4 for _ in range(len(g_idx1)))
    obs_tensor[g_idx1, g_idx2, g_idx3] = 1.0

    return obs_tensor


def my_nature_cnn(input_shape, **conv_kwargs):
    """
    CNN from Nature paper.
    """
    print('input shape is {}'.format(input_shape))
    x_input = tf.keras.Input(shape=input_shape, dtype=tf.float32)
    h = x_input
    h = conv('c1', nf=8, rf=4, stride=1, activation='relu', init_scale=np.sqrt(2))(h)
    h = tf.keras.layers.Flatten()(h)
    h = tf.keras.layers.Dense(units=512, kernel_initializer=ortho_init(np.sqrt(2)),
                              name='fc1', activation='relu')(h)
    network = tf.keras.Model(inputs=[x_input], outputs=[h])
    return network


@register("my_cnn")
def my_cnn(**conv_kwargs):
    def network_fn(input_shape):
        return my_nature_cnn(input_shape, **conv_kwargs)
    return network_fn
