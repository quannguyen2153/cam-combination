from typing import List
import numpy as np
import torch
import tqdm
from pytorch_grad_cam.base_cam import BaseCAM
from pytorch_grad_cam.utils.find_layers import find_layer_predicate_recursive
from pytorch_grad_cam.utils.image import scale_accross_batch_and_channels
from pytorch_grad_cam.utils.svd_on_activations import get_2d_projection

class FullUnifiedCAM(BaseCAM):
    def __init__(self, model, target_layers, reshape_transform=None):
        if not isinstance(target_layers, list) or len(target_layers) <= 0:
            print("INFO: All bias layers will be used")

            def layer_with_2D_bias(layer):
                bias_target_layers = [torch.nn.Conv2d, torch.nn.BatchNorm2d]
                if type(layer) in bias_target_layers and layer.bias is not None:
                    return True
                return False
            
            target_layers = find_layer_predicate_recursive(
                model, layer_with_2D_bias)
            
            print(f"INFO: {len(target_layers)} bias layers will be accounted for.")

        super(FullUnifiedCAM, self).__init__(model=model, target_layers=target_layers, reshape_transform=reshape_transform, compute_input_gradient=True)

    def forward(
        self, input_tensor: torch.Tensor, targets: List[torch.nn.Module], eigen_smooth: bool = False
    ) -> np.ndarray:
        input_tensor = input_tensor.to(self.device)

        if self.compute_input_gradient:
            input_tensor = torch.autograd.Variable(input_tensor, requires_grad=True)

        self.outputs = outputs = self.activations_and_grads(input_tensor)
        self.activations_and_grads.release() # Release hooks to avoid accumulating memory size when computing

        if self.uses_gradients:
            self.model.zero_grad()
            loss = sum([target(output) for target, output in zip(targets, outputs)])
            loss.backward(retain_graph=True)

        cam_per_layer = self.compute_cam_per_layer(input_tensor, targets, eigen_smooth)
        return self.aggregate_multi_layers(cam_per_layer)
        
    def get_bias_data(self, layer):
        if isinstance(layer, torch.nn.BatchNorm2d):
            bias = - (layer.running_mean * layer.weight / torch.sqrt(layer.running_var + layer.eps)) + layer.bias
            return bias
        else:
            return layer.bias
        
    def get_cam_image(
        self,
        input_tensor: torch.Tensor,
        target_layer: torch.nn.Module,
        targets: List[torch.nn.Module],
        activations: torch.Tensor,
        layer_grads: torch.Tensor,
        eigen_smooth: bool = False,
    ) -> np.ndarray:
        # Calculate filtering weights using GradCAM++ approach
        grads_power_2 = layer_grads**2
        grads_power_3 = grads_power_2 * layer_grads
        # Equation 19 in https://arxiv.org/abs/1710.11063
        sum_activations = np.sum(activations, axis=(2, 3))
        eps = 0.000001
        aij = grads_power_2 / (2 * grads_power_2 +
                               sum_activations[:, :, None, None] * grads_power_3 + eps)
        # Now bring back the ReLU from eq.7 in the paper,
        # And zero out aijs where the activations are 0
        aij = np.where(layer_grads != 0, aij, 0)

        grad_cam_weights = np.maximum(layer_grads, 0) * aij
        grad_cam_weights = np.sum(grad_cam_weights, axis=(2, 3))

        mask = grad_cam_weights > -99999999

        with torch.no_grad():
            # Filter activations using the mask
            filtered_activations = np.stack([
                activations[batch_idx][mask[batch_idx]] for batch_idx in range(activations.shape[0])
            ])

            # Filter and normalize layer grads
            filtered_layer_grads = np.stack([
                layer_grads[batch_idx][mask[batch_idx]] for batch_idx in range(layer_grads.shape[0])
            ])
            eps = 1e-7
            sum_activations = np.sum(filtered_activations, axis=(2, 3))
            normalized_layer_grads = torch.nn.Softmax(dim=-1)(torch.from_numpy(filtered_layer_grads * filtered_activations / (sum_activations[:, :, None, None] + eps))).to(self.device)

            # Filter and normalize biases
            try:
                biases = self.get_bias_data(target_layer)

                filtered_biases = []
                for batch_idx in range(mask.shape[0]):
                    filtered_biases.append(biases[mask[batch_idx]])
                filtered_biases = torch.stack(filtered_biases)[:, :, None, None]

            except:
                # If the layer doesn't have bias
                filtered_biases = torch.zeros(filtered_activations.shape, device=self.device)

            # Highlight important pixels in each activation
            modified_activations = normalized_layer_grads * torch.from_numpy(filtered_activations).to(self.device)

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
            heat_maps = input_tensor[:, None, :, :] * upsampled[:, :, None, :, :]

            # Append input * gradient map
            target_size = self.get_target_width_height(input_tensor)
            input_grad = input_tensor.grad.data.cpu().numpy()
            gradient_multiplied_input = input_grad * input_tensor.data.cpu().numpy()
            gradient_multiplied_input = np.abs(gradient_multiplied_input)
            gradient_multiplied_input = torch.from_numpy(scale_accross_batch_and_channels(gradient_multiplied_input, target_size)).to(self.device)

            # Append bias * gradient maps
            gradient_multiplied_biases = np.abs(filtered_biases.cpu().numpy() * filtered_layer_grads)
            gradient_multiplied_biases = torch.from_numpy(scale_accross_batch_and_channels(gradient_multiplied_biases, target_size)).to(self.device)
            
            
            # Concantenate feature maps
            feature_maps = torch.cat([upsampled, gradient_multiplied_biases], dim=1)

            # Concatenate heat maps
            heat_maps = torch.cat([heat_maps, gradient_multiplied_biases[:, :, None, :, :] * input_tensor[:, None, :, :]], dim=1)

            if hasattr(self, "batch_size"):
                BATCH_SIZE = self.batch_size
            else:
                BATCH_SIZE = 16

            # Calculate score of each pertubated inputs
            scores = []
            for target, tensor in zip(targets, heat_maps):
                for i in tqdm.tqdm(range(0, tensor.size(0), BATCH_SIZE)):
                    batch = tensor[i: i + BATCH_SIZE, :]
                    outputs = [target(o).to(self.device).item()
                               for o in self.model(batch)]
                    scores.extend(outputs)
            scores = torch.Tensor(scores)
            scores = scores.view(heat_maps.shape[0], heat_maps.shape[1])
            weights = torch.nn.Softmax(dim=-1)(scores).numpy()

            # Aggregate activations to saliency map
            feature_maps = feature_maps.cpu().numpy()

            # 2D conv
            if len(feature_maps.shape) == 4:
                weighted_activations = weights[:, :, None, None] * feature_maps
            # 3D conv
            elif len(feature_maps.shape) == 5:
                weighted_activations = weights[:, :, None, None, None] * feature_maps
            else:
                raise ValueError(f"Invalid activation shape. Get {len(feature_maps.shape)}.")

            if eigen_smooth:
                cam = get_2d_projection(weighted_activations)
            else:
                cam = weighted_activations.sum(axis=1)
            return cam
