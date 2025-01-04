from typing import List
import numpy as np
import torch
import tqdm
from pytorch_grad_cam.base_cam import BaseCAM
from pytorch_grad_cam.utils.svd_on_activations import get_2d_projection

class UnifiedCAM(BaseCAM):
    def __init__(self, model, target_layers,
                 reshape_transform=None):
        super(UnifiedCAM, self).__init__(model, target_layers,
                                              reshape_transform)
        
    def get_cam_image(
        self,
        input_tensor: torch.Tensor,
        target_layer: torch.nn.Module,
        targets: List[torch.nn.Module],
        activations: torch.Tensor,
        grads: torch.Tensor,
        eigen_smooth: bool = False,
    ) -> np.ndarray:
        grads_power_2 = grads**2
        grads_power_3 = grads_power_2 * grads
        # Equation 19 in https://arxiv.org/abs/1710.11063
        sum_activations = np.sum(activations, axis=(2, 3))
        eps = 0.000001
        aij = grads_power_2 / (2 * grads_power_2 +
                               sum_activations[:, :, None, None] * grads_power_3 + eps)
        # Now bring back the ReLU from eq.7 in the paper,
        # And zero out aijs where the activations are 0
        aij = np.where(grads != 0, aij, 0)

        grad_cam_weights = np.maximum(grads, 0) * aij
        grad_cam_weights = np.sum(grad_cam_weights, axis=(2, 3))

        mask = grad_cam_weights > 0

        with torch.no_grad():
            # Filter activations using the mask
            filtered_activations = torch.tensor(np.stack([
                activations[batch_idx][mask[batch_idx]] for batch_idx in range(activations.shape[0])
            ])).to(self.device)

            # Normalize and filter grads
            normalized_grads = torch.nn.Softmax(dim=-1)(torch.from_numpy(grads)).cpu().numpy()
            filtered_grads = torch.tensor(np.stack([
                normalized_grads[batch_idx][mask[batch_idx]] for batch_idx in range(normalized_grads.shape[0])
            ])).to(self.device)

            modified_activations = filtered_activations * filtered_grads

            upsample = torch.nn.UpsamplingBilinear2d(size=input_tensor.shape[-2:])
            upsampled = upsample(modified_activations)

            maxs = upsampled.view(upsampled.size(0),
                                    upsampled.size(1), -1).max(dim=-1)[0]
            mins = upsampled.view(upsampled.size(0),
                                    upsampled.size(1), -1).min(dim=-1)[0]

            maxs, mins = maxs[:, :, None, None], mins[:, :, None, None]
            upsampled = (upsampled - mins) / (maxs - mins + 1e-8)

            input_tensors = input_tensor[:, None,
                                            :, :] * upsampled[:, :, None, :, :]

            if hasattr(self, "batch_size"):
                BATCH_SIZE = self.batch_size
            else:
                BATCH_SIZE = 16

            scores = []
            for target, tensor in zip(targets, input_tensors):
                for i in tqdm.tqdm(range(0, tensor.size(0), BATCH_SIZE)):
                    batch = tensor[i: i + BATCH_SIZE, :]
                    outputs = [target(o).to(self.device).item()
                               for o in self.model(batch)]
                    scores.extend(outputs)
            scores = torch.Tensor(scores)
            scores = scores.view(modified_activations.shape[0], modified_activations.shape[1])
            weights = torch.nn.Softmax(dim=-1)(scores).numpy()

            modified_activations = modified_activations.cpu().numpy()

            # 2D conv
            if len(modified_activations.shape) == 4:
                weighted_activations = weights[:, :, None, None] * modified_activations
            # 3D conv
            elif len(modified_activations.shape) == 5:
                weighted_activations = weights[:, :, None, None, None] * modified_activations
            else:
                raise ValueError(f"Invalid activation shape. Get {len(modified_activations.shape)}.")

            if eigen_smooth:
                cam = get_2d_projection(weighted_activations)
            else:
                cam = weighted_activations.sum(axis=1)
            return cam
