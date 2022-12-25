import jax
import jax.numpy as jnp
from flax import linen as nn


class Cnn(nn.Module):
    @nn.compact
    def __call__(self, x, is_training):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.BatchNorm(use_running_average=not is_training, momentum=0.1)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.BatchNorm(use_running_average=not is_training, momentum=0.1)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        x = nn.log_softmax(x)

        return x


if __name__ == "__main__":
    model = Cnn()
    key1, key2 = jax.random.split(jax.random.PRNGKey(1))
    dummy_input = jnp.ones((10, 28, 28, 1))
    params = model.init(key1, dummy_input, True)
