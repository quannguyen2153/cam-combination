from typing import List, Callable
import numpy as np
import torch

class PerturbationConfidenceMetric:
    def __init__(self, perturbation):
        self.perturbation = perturbation

    def __call__(self, input_tensor: torch.Tensor,
                 cams: np.ndarray,
                 targets: List[Callable],
                 model: torch.nn.Module,
                 return_visualization=False,
                 return_diff=True):

        if return_diff:
            with torch.no_grad():
                outputs = model(input_tensor)
                scores = [target(output).cpu().numpy()
                          for target, output in zip(targets, outputs)]
                scores = np.float32(scores)

        batch_size = input_tensor.size(0)
        perturbated_tensors = []
        for i in range(batch_size):
            cam = cams[i]
            tensor = self.perturbation(input_tensor[i, ...].cpu(),
                                       torch.from_numpy(cam))
            tensor = tensor.to(input_tensor.device)
            perturbated_tensors.append(tensor.unsqueeze(0))
        perturbated_tensors = torch.cat(perturbated_tensors)

        with torch.no_grad():
            outputs_after_imputation = model(perturbated_tensors)
        scores_after_imputation = [
            target(output).cpu().numpy() for target, output in zip(
                targets, outputs_after_imputation)]
        scores_after_imputation = np.float32(scores_after_imputation)

        if return_diff:
            result = scores_after_imputation - scores
        else:
            result = scores_after_imputation

        if return_visualization:
            return result, scores, scores_after_imputation, perturbated_tensors
        else:
            return result, scores, scores_after_imputation

def multiply_tensor_with_cam(input_tensor: torch.Tensor,
                             cam: torch.Tensor):
    """ Multiply an input tensor (after normalization)
        with a pixel attribution map
    """
    return input_tensor * cam
        
class CamMultImageConfidenceChange(PerturbationConfidenceMetric):
    def __init__(self):
        super(CamMultImageConfidenceChange,
              self).__init__(multiply_tensor_with_cam)
        
class DropInConfidence(CamMultImageConfidenceChange):
    def __init__(self):
        super(DropInConfidence, self).__init__()

    def __call__(self, *args, **kwargs):
        scores, scores_before, scores_after = super(DropInConfidence, self).__call__(*args, **kwargs)
        scores = -scores
        return np.maximum(scores, 0) / scores_before * 100


class IncreaseInConfidence(CamMultImageConfidenceChange):
    def __init__(self):
        super(IncreaseInConfidence, self).__init__()

    def __call__(self, *args, **kwargs):
        scores, bef_score, scores_after = super(IncreaseInConfidence, self).__call__(*args, **kwargs)
        return np.float32(scores > 0)