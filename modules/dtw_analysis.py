import numpy as np

from librosa.sequence import dtw

# =========================================================
# MODULE 1
# Dynamic Time Warping
# =========================================================

def compute_dtw(system_mfcc, ground_truth_mfcc):

    """
    Computes DTW alignment between two MFCC matrices.

    Parameters
    ----------
    system_mfcc : numpy.ndarray

    ground_truth_mfcc : numpy.ndarray

    Returns
    -------
    dict
    """

    cost_matrix, alignment_path = dtw(

        X=system_mfcc,

        Y=ground_truth_mfcc,

        metric="euclidean"

    )

    distance = float(cost_matrix[-1, -1])

    path_length = len(alignment_path)

    normalized_distance = distance / path_length

    similarity = 1 / (1 + normalized_distance)

    return {

        "distance": distance,

        "normalized_distance": normalized_distance,

        "path_length": path_length,

        "similarity": similarity,

        "cost_matrix": cost_matrix,

        "alignment_path": alignment_path

    }

