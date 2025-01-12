from typing import List
import numpy as np
import torch
import tqdm
from pytorch_grad_cam.base_cam import BaseCAM
from pytorch_grad_cam.utils.find_layers import find_layer_predicate_recursive
from pytorch_grad_cam.utils.svd_on_activations import get_2d_projection

class UnifiedCAM(BaseCAM):
    def __init__(self, model, target_layers,
                 reshape_transform=None):
        if not isinstance(target_layers, list) or len(target_layers) <= 0:
            print("INFO: All bias layers will be used")

            def layer_with_2D_bias(layer):
                bias_target_layers = [torch.nn.Conv2d, torch.nn.BatchNorm2d]
                if type(layer) in bias_target_layers and layer.bias is not None:
                    return True
                return False
            
            target_layers = find_layer_predicate_recursive(
                model, layer_with_2D_bias)
            
            print(f"{len(target_layers)} bias layers will be accounted for.")

        super(UnifiedCAM, self).__init__(model, target_layers, reshape_transform)

    def forward(
        self, input_tensor: torch.Tensor, targets: List[torch.nn.Module], eigen_smooth: bool = False
    ) -> np.ndarray:
        input_tensor = input_tensor.to(self.device)

        if self.compute_input_gradient:
            input_tensor = torch.autograd.Variable(input_tensor, requires_grad=True)

        self.outputs = outputs = self.activations_and_grads(input_tensor)
        self.activations_and_grads.release() # Release hooks to avoid accumulating memory size when computing

        if targets is None:
            target_categories = np.argmax(outputs.cpu().data.numpy(), axis=-1)
            targets = [ClassifierOutputTarget(category) for category in target_categories]

        if self.uses_gradients:
            self.model.zero_grad()
            loss = sum([target(output) for target, output in zip(targets, outputs)])
            loss.backward(retain_graph=True)

        cam_per_layer = self.compute_cam_per_layer(input_tensor, targets, eigen_smooth)
        return self.aggregate_multi_layers(cam_per_layer)
        
    def get_bias_data(self, layer):
        # Borrowed from official paper impl:
        # https://github.com/idiap/fullgrad-saliency/blob/master/saliency/tensor_extractor.py#L47
        if isinstance(layer, torch.nn.BatchNorm2d):
            bias = - (layer.running_mean * layer.weight
                      / torch.sqrt(layer.running_var + layer.eps)) + layer.bias
            return bias
        else:
            return layer.bias
        
    def get_cam_image(
        self,
        input_tensor: torch.Tensor,
        target_layer: torch.nn.Module,
        targets: List[torch.nn.Module],
        activations: torch.Tensor,
        grads: torch.Tensor,
        eigen_smooth: bool = False,
    ) -> np.ndarray:
        # Calculate filtering weights using GradCAM++ approach
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
            filtered_activations = np.stack([
                activations[batch_idx][mask[batch_idx]] for batch_idx in range(activations.shape[0])
            ])

            # Filter and normalize grads
            filtered_grads = np.stack([
                grads[batch_idx][mask[batch_idx]] for batch_idx in range(grads.shape[0])
            ])
            eps = 1e-7
            sum_activations = np.sum(filtered_activations, axis=(2, 3))
            normalized_grads = torch.nn.Softmax(dim=-1)(torch.from_numpy(filtered_grads * filtered_activations / (sum_activations[:, :, None, None] + eps))).to(self.device)

            # Filter and normalize biases
            biases = self.get_bias_data(target_layer)
            filtered_biases = []
            for batch_idx in range(mask.shape[0]):
                filtered_biases.append(biases[mask[batch_idx]])
            filtered_biases = torch.stack(filtered_biases)

            min_biases = filtered_biases.min(dim=-1, keepdim=True)[0]
            max_biases = filtered_biases.max(dim=-1, keepdim=True)[0]
            normalized_biases = (filtered_biases - min_biases) / (max_biases - min_biases + 1e-8)
            normalized_biases = normalized_biases[:, :, None, None]

            # Highlight important pixels in each activation
            modified_activations = torch.from_numpy(filtered_activations).to(self.device) * normalized_grads * normalized_biases

            # Upsample and normalize activations
            upsample = torch.nn.UpsamplingBilinear2d(size=input_tensor.shape[-2:])
            upsampled = upsample(modified_activations)

            maxs = upsampled.view(upsampled.size(0),
                                    upsampled.size(1), -1).max(dim=-1)[0]
            mins = upsampled.view(upsampled.size(0),
                                    upsampled.size(1), -1).min(dim=-1)[0]

            maxs, mins = maxs[:, :, None, None], mins[:, :, None, None]
            upsampled = (upsampled - mins) / (maxs - mins + 1e-8)

            # Pertubate input with each activation
            input_tensors = input_tensor[:, None,
                                            :, :] * upsampled[:, :, None, :, :]

            if hasattr(self, "batch_size"):
                BATCH_SIZE = self.batch_size
            else:
                BATCH_SIZE = 16

            # Calculate score of each pertubated inputs
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

            # Aggregate activations to saliency map
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
