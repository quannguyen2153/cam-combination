from typing import List
import numpy as np

def filter_grayscale(grayscale: np.ndarray, threshold: float) -> np.ndarray:
    threshold_value = np.max(grayscale) * threshold

    mask = grayscale >= threshold_value

    filtered_grayscale = grayscale.copy()
    filtered_grayscale[~mask] = 0

    return filtered_grayscale

def combine_by_weight(grayscales: List[np.ndarray], weights: List[float]) -> np.ndarray:
    if sum(weights) != 1:
        raise ValueError("sum of weights must be 1")
    if len(grayscales) != len(weights):
        raise ValueError("length of grayscales and weights must be equal")
    shape = grayscales[0].shape
    for grayscale in grayscales:
        if grayscale.shape != shape:
            raise ValueError("shape of all grayscales must be the same")
    
    combination = np.zeros_like(grayscales[0])

    for grayscale, weight in zip(grayscales, weights):
        combination += grayscale * weight

    return combination

def combine_by_matching_important_pixels(grayscales: List[np.ndarray], thresholds: List[float]) -> np.ndarray:
    if len(grayscales) != len(thresholds):
        raise ValueError("length of grayscales and thresholds must be equal")
    shape = grayscales[0].shape
    for grayscale in grayscales:
        if grayscale.shape != shape:
            raise ValueError("shape of all grayscales must be the same")
        
    filtered_grayscales = []
    for grayscale, threshold in zip(grayscales, thresholds):
        filtered_grayscale = grayscale.copy()
        percentile_value = np.percentile(filtered_grayscale, threshold)
        filtered_grayscale[filtered_grayscale < percentile_value] = 0
        filtered_grayscales.append(filtered_grayscale)

    combination = np.zeros_like(filtered_grayscales[0])

    for i in range(shape[0]):
        for j in range(shape[1]):
            max_value = 0
            for grayscale in filtered_grayscales:
                if grayscale[i, j] != 0:
                    max_value = max(max_value, grayscale[i, j])
                else:
                    max_value = 0
                    break
            combination[i, j] = max_value

    return combination