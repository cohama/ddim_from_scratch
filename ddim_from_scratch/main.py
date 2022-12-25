from datetime import datetime
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow as tf
from flax.training.train_state import TrainState
from tqdm import tqdm

import tensorflow_datasets as tfds
from ddim_from_scratch.model import DiffusionModel


class BatchGen:
    def __init__(self, batch_size: int, input_image_height: int, input_image_width: int):
        self._batch_size = batch_size
        self.input_image_width = input_image_width
        self.input_image_height = input_image_height

        train_ds = tfds.load(  # type: ignore[attr-defined]
            name="oxford_flowers102", shuffle_files=True, split="train[:80%]+validation[:80%]+test[:80%]"
        )
        self.train_set = (
            train_ds.shuffle(len(train_ds), reshuffle_each_iteration=True)
            .map(self.preprocessing)
            .batch(batch_size)
            .prefetch(10)
        )
        val_ds = tfds.load(  # type: ignore[attr-defined]
            name="oxford_flowers102", shuffle_files=True, split="train[80%:]+validation[80%:]+test[80%:]"
        )
        self.test_set = val_ds.map(self.preprocessing).batch(batch_size).prefetch(10)

        train_samples = np.concatenate(list(self.train_set.take(100).as_numpy_iterator()))
        train_samples = train_samples.reshape((-1, 3))
        self.dataset_mean = train_samples.mean(axis=0)
        self.dataset_std = train_samples.std(axis=0)

    def preprocessing(self, data):
        shape = tf.shape(data["image"])
        height = shape[0]
        width = shape[1]
        crop_size = tf.minimum(height, width)
        image = tf.image.crop_to_bounding_box(
            data["image"],
            (height - crop_size) // 2,
            (width - crop_size) // 2,
            crop_size,
            crop_size,
        )
        image = tf.image.resize(image, size=[self.input_image_height, self.input_image_width], antialias=True)
        return tf.clip_by_value(image / 255, 0.0, 1.0)


class MyTrainState(TrainState):
    batch_stats: dict


@partial(jax.jit, static_argnums=(3,))
def train_step(images: jax.Array, rng: jax.random.KeyArray, state: MyTrainState, is_training=True):
    def loss_fn(params, batch_stats):
        (pred_noises, pred_images, noises, noisy_images), mutated_vars = state.apply_fn(
            {
                "params": params,
                "batch_stats": batch_stats,
            },
            images,
            rng,
            mutable=["batch_stats"],
        )
        noise_loss = jnp.abs(noises - pred_noises).mean()
        image_loss = jnp.abs(images - pred_images).mean()
        return noise_loss, (image_loss, pred_images, noisy_images, mutated_vars)

    if is_training:
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (noise_loss, (image_loss, pred_images, noisy_images, mutated_vars)), grads = grad_fn(
            state.params, state.batch_stats
        )
        state = state.apply_gradients(grads=grads, batch_stats=mutated_vars["batch_stats"])
    else:
        noise_loss, (image_loss, pred_images, noisy_images, mutated_vars) = loss_fn(state.params, state.batch_stats)
    return noise_loss, image_loss, pred_images, noisy_images, state


def _update_ema(p_cur, p_new, momentum: float = 0.999):
    return momentum * p_cur + (1 - momentum) * p_new


def run(
    batch_size: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    log_dir: Path,
):
    batch_gen = BatchGen(batch_size=32, input_image_height=32, input_image_width=32)
    model = DiffusionModel(
        max_signal_rates=0.95,
        min_signal_rates=0.02,
        image_width=32,
        image_height=32,
        dataset_mean=batch_gen.dataset_mean,
        dataset_std=batch_gen.dataset_std,
    )
    key1, rng = jax.random.split(jax.random.PRNGKey(1))
    key2, rng = jax.random.split(rng)
    dummy_input = jnp.ones((batch_size, batch_gen.input_image_height, batch_gen.input_image_width, 3))
    variables = model.init(key1, dummy_input, key2)

    params = variables["params"]
    batch_stats = variables["batch_stats"]

    state = MyTrainState.create(
        apply_fn=model.apply,
        params=params,
        batch_stats=batch_stats,
        tx=optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay),
    )
    ema_params = state.params.copy(add_or_replace={})

    summary_writer = tf.summary.create_file_writer(str(log_dir / datetime.now().strftime("%Y%m%d-%H%M%S")))
    for epoch in tqdm(range(epochs)):
        train_noise_loss, train_image_loss, _test_noise_loss, _test_image_loss = 0.0, 0.0, 0.0, 0.0

        for images in batch_gen.train_set.as_numpy_iterator():
            key_train, rng = jax.random.split(rng)
            noise_loss, image_loss, pred_images, noisy_images, state = train_step(
                images, key_train, state, is_training=True
            )
            train_noise_loss += noise_loss
            train_image_loss += image_loss
            ema_params = jax.tree_map(_update_ema, ema_params, state.params)

        train_noise_loss /= float(len(batch_gen.train_set))
        train_image_loss /= float(len(batch_gen.train_set))

        eval_key, rng = jax.random.split(rng)
        diff_key, rng = jax.random.split(rng)
        generated_iamges = model.apply(
            variables={
                "params": ema_params,
                "batch_stats": state.batch_stats,
            },
            method=model.generate,
            num_images=10,
            diffusion_steps=20,
            rng=diff_key,
        )

        with summary_writer.as_default(step=epoch):
            tf.summary.scalar("train_noise_loss", train_noise_loss)
            tf.summary.scalar("train_image_loss", train_image_loss)
            tf.summary.image("generated_iamge", generated_iamges, max_outputs=10)
            tf.summary.image("train_images", images, max_outputs=10)
            tf.summary.image("train_noisy_images", noisy_images, max_outputs=10)
            tf.summary.image("train_denoised_images", pred_images, max_outputs=10)


if __name__ == "__main__":
    run(
        batch_size=64,
        epochs=50,
        learning_rate=1e-3,
        weight_decay=1e-4,
        log_dir=Path("log"),
    )
