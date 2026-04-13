import numpy as np


def apply_crossfade(prev_tail: np.ndarray, chunk: np.ndarray) -> np.ndarray:
    """Blend the tail of the previous chunk into the head of the current one."""
    n = len(prev_tail)
    if len(chunk) < n:
        return chunk  # Chunk too short, skip crossfade

    fade_out = np.linspace(1.0, 0.0, n, dtype="float32")
    fade_in = np.linspace(0.0, 1.0, n, dtype="float32")

    result = chunk.copy()
    result[:n] = prev_tail * fade_out + chunk[:n] * fade_in
    return result


def trim_silence(chunk: np.ndarray, sample_rate: int, silence_window_ms: int, silence_rms_threshold: float) -> np.ndarray:
    """Trim leading and trailing silence from a chunk based on an RMS threshold."""
    window_samples = int(sample_rate * silence_window_ms / 1000)

    # Compute RMS over sliding windows
    num_windows = len(chunk) // window_samples
    if num_windows == 0:
        return chunk

    windows = chunk[:num_windows * window_samples].reshape(num_windows, window_samples)
    rms = np.sqrt(np.mean(windows ** 2, axis=1))

    # Find first and last window above the threshold
    active = np.where(rms > silence_rms_threshold)[0]
    if len(active) == 0:
        return chunk  # Fully silent chunk, return as-is to avoid empty array

    start = active[0] * window_samples
    end   = (active[-1] + 1) * window_samples

    return chunk[start:end]