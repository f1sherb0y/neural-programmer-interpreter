import tensorflow as tf


def configure_tensorflow(
    device: str = "auto",
    *,
    memory_growth: bool = True,
    mixed_precision: bool = False,
) -> str:
    gpus = tf.config.list_physical_devices("GPU")
    if device == "cpu":
        tf.config.set_visible_devices([], "GPU")
        selected = "CPU"
    elif device == "auto":
        selected = gpus[0].name if gpus else "CPU"
    else:
        index = int(device.removeprefix("gpu:"))
        if index >= len(gpus):
            raise RuntimeError(
                f"Requested GPU {index}, but TensorFlow sees {len(gpus)}"
            )
        tf.config.set_visible_devices([gpus[index]], "GPU")
        gpus = [gpus[index]]
        selected = gpus[0].name
    if memory_growth:
        for gpu in tf.config.get_visible_devices("GPU"):
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass
    if mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_bfloat16")
    return selected


def configure_worker_gpu(gpu_id: int, memory_growth: bool = True) -> str:
    gpus = tf.config.list_physical_devices("GPU")
    if gpu_id >= len(gpus):
        raise RuntimeError(f"Requested GPU {gpu_id}, but TensorFlow sees {len(gpus)}")
    gpu = gpus[gpu_id]
    tf.config.set_visible_devices([gpu], "GPU")
    if memory_growth:
        tf.config.experimental.set_memory_growth(gpu, True)
    return gpu.name
