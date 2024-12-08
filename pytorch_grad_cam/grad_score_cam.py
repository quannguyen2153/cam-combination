import numpy as np
import torch
import tqdm
from pytorch_grad_cam.base_cam import BaseCAM

class GradScoreCAM(BaseCAM):
    def __init__(self, model, target_layers,
                 reshape_transform=None):
        super(GradScoreCAM, self).__init__(model, target_layers,
                                              reshape_transform)

    def get_cam_weights(self,
                        input_tensor,
                        target_layers,
                        targets,
                        activations,
                        grads):
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
            filtered_activations = torch.tensor(np.stack([
                activations[batch_idx][mask[batch_idx]] for batch_idx in range(activations.shape[0])
            ])).to(self.device)

            upsample = torch.nn.UpsamplingBilinear2d(size=input_tensor.shape[-2:])
            upsampled = upsample(filtered_activations)

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

            scores = torch.full(mask.shape, torch.finfo(torch.float16).min).to(self.device)
            for batch_idx, (target, tensor) in enumerate(zip(targets, input_tensors)):
                batch_scores = []
                for i in tqdm.tqdm(range(0, tensor.size(0), BATCH_SIZE)):
                    batch = tensor[i: i + BATCH_SIZE, :]
                    outputs = [target(o).to(self.device) for o in self.model(batch)]
                    batch_scores.extend(outputs)

                batch_scores = torch.cat(batch_scores, dim=0)
                scores[batch_idx][mask[batch_idx]] = batch_scores

            weights = torch.nn.Softmax(dim=-1)(scores).cpu().numpy()
            return weights
