"""Kernel-agnostic correctness comparison utilities."""
import numpy as np


def cos_sim(a, b):
    a = np.asarray(a).flatten().astype(np.float32)
    b = np.asarray(b).flatten().astype(np.float32)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def max_diff(a, b):
    return float(np.max(np.abs(
        np.asarray(a).astype(np.float32) - np.asarray(b).astype(np.float32)
    )))


def mean_diff(a, b):
    return float(np.mean(np.abs(
        np.asarray(a).astype(np.float32) - np.asarray(b).astype(np.float32)
    )))


def check_pass(cos, threshold=0.9999):
    if cos > threshold:
        return "PASS"
    elif cos > 0.99:
        return "MARGINAL"
    else:
        return "FAIL"
