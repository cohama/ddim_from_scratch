from typing import Any

import jax
import jax.numpy as jnp
from flax import linen as nn


class SinusoidalEmbedding(nn.Module):
    embedding_dims: int

    @nn.compact
    def __call__(self, x):
        embedding_min_frequency = 1.0
        embedding_max_frequency = 1000.0
        frequencies = jnp.exp(
            jnp.linspace(
                jnp.log(embedding_min_frequency),
                jnp.log(embedding_max_frequency),
                self.embedding_dims // 2,
            )
        )
        angular_speeds = 2.0 * jnp.pi * frequencies
        return jnp.concatenate([jnp.sin(angular_speeds * x), jnp.cos(angular_speeds * x)], axis=3)


class ConvBlock(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x, is_training: bool) -> Any:
        x = nn.Conv(features=self.features, kernel_size=(1, 1))(x)
        sub_x = x
        x = nn.Conv(features=self.features, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.BatchNorm(
            use_running_average=not is_training,
            use_bias=False,
            use_scale=False,
        )(x)
        x = nn.relu(x)
        x = nn.Conv(features=self.features, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.BatchNorm(
            use_running_average=not is_training,
            use_bias=False,
            use_scale=False,
        )(x)
        x = nn.relu(x)
        return sub_x + x


class UpConvBlock(nn.Module):
    features: int


def downsample(x) -> Any:
    return nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2), padding="SAME")


def upsample(x) -> Any:
    b, h, w, c = x.shape
    return jax.image.resize(x, (b, h * 2, w * 2, c), method="bicubic")


class UNetLikeModel(nn.Module):
    @nn.compact
    def __call__(self, noisy_images, noisy_variances, is_training: bool) -> Any:
        emb = SinusoidalEmbedding(32)(noisy_variances)
        _b, image_height, image_width, _c = noisy_images.shape
        emb = jnp.tile(emb, (1, image_height, image_width, 1))

        x = nn.Conv(32, (1, 1))(noisy_images)
        f1 = jnp.concatenate([x, emb], axis=-1)
        f1 = ConvBlock(32)(f1, is_training)

        f2 = ConvBlock(64)(downsample(f1), is_training)
        f2 = ConvBlock(64)(f2, is_training)

        f4 = ConvBlock(96)(downsample(f2), is_training)
        f4 = ConvBlock(96)(f4, is_training)

        f8 = ConvBlock(128)(downsample(f4), is_training)
        f8 = ConvBlock(128)(f8, is_training)

        p4 = ConvBlock(96)(jnp.concatenate([f4, upsample(f8)], axis=-1), is_training)
        p4 = ConvBlock(96)(p4, is_training)

        p2 = ConvBlock(64)(jnp.concatenate([f2, upsample(p4)], axis=-1), is_training)
        p2 = ConvBlock(64)(p2, is_training)

        p1 = ConvBlock(32)(jnp.concatenate([f1, upsample(p2)], axis=-1), is_training)
        p1 = ConvBlock(32)(p1, is_training)

        return nn.Conv(3, (1, 1), kernel_init=nn.initializers.zeros)(p1)


class DiffusionModel(nn.Module):
    max_signal_rates: float
    min_signal_rates: float

    image_width: int
    image_height: int

    dataset_mean: jax.Array
    dataset_std: jax.Array

    def setup(self):
        self.network = UNetLikeModel()

    def _diffusion_schedule(self, diffusion_times: jax.Array) -> tuple[jax.Array, jax.Array]:
        # diffusion times -> angles
        start_angle = jnp.arccos(self.max_signal_rates)
        end_angle = jnp.arccos(self.min_signal_rates)

        diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

        # angles -> signal and noise raites
        signal_rates = jnp.cos(diffusion_angles)
        noise_rates = jnp.sin(diffusion_angles)

        return noise_rates, signal_rates

    def _denoise(self, noisy_images, noise_rates, signal_rates, is_training: bool) -> tuple[jax.Array, jax.Array]:
        pred_noises = self.network(noisy_images, noise_rates**2, is_training)
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates

        return pred_noises, pred_images

    def _reverse_diffusion(self, initial_noise, diffusion_steps: int) -> jax.Array:
        num_images = initial_noise.shape[0]
        step_size = 1.0 / diffusion_steps

        next_noisy_images = initial_noise
        for step in range(diffusion_steps):
            noisy_images = next_noisy_images

            diffusion_times = jnp.ones((num_images, 1, 1, 1)) - step * step_size
            noise_rates, signal_rates = self._diffusion_schedule(diffusion_times)
            pred_noises, pred_images = self._denoise(noisy_images, noise_rates, signal_rates, is_training=False)

            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self._diffusion_schedule(next_diffusion_times)
            next_noisy_images = next_signal_rates * pred_images + next_noise_rates * pred_noises

        return pred_images

    def _normalize(self, x: jax.Array) -> jax.Array:
        return (x - self.dataset_mean) / (self.dataset_std + 1e-5)

    def _denormalize(self, x: jax.Array) -> jax.Array:
        return self.dataset_std * x + self.dataset_mean

    def generate(self, num_images: int, diffusion_steps: int, rng: jax.random.KeyArray) -> jax.Array:
        """ノイズから画像を生成する"""
        initial_noise = jax.random.normal(rng, shape=(num_images, self.image_height, self.image_width, 3))
        generate_images = self._reverse_diffusion(initial_noise, diffusion_steps)
        return self._denormalize(generate_images)

    def __call__(
        self, images: jax.Array, rng: jax.random.KeyArray
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """訓練用にノイズ付与された画像をデノイズする

        Args:
            images: 訓練用の入力画像 shape(B, H, W, 3)
            rng: 乱数

        Returns:
            pred_noises, pred_images
        """
        normed_images = self._normalize(images)
        rng_noise, rng = jax.random.split(rng)
        noises = jax.random.normal(key=rng_noise, shape=images.shape)
        rng_time, rng = jax.random.split(rng)
        diffusion_times = jax.random.uniform(key=rng_time, shape=(images.shape[0], 1, 1, 1), minval=0.0, maxval=1.0)
        noise_rates, signal_rates = self._diffusion_schedule(diffusion_times)
        noisy_images = signal_rates * normed_images + noise_rates * noises
        pred_noises, pred_images = self._denoise(noisy_images, noise_rates, signal_rates, is_training=True)
        pred_images = self._denormalize(pred_images)
        return pred_noises, pred_images, noises, noisy_images
