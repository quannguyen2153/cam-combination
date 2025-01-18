from typing import List
import numpy as np
import torch
import tqdm
from pytorch_grad_cam.base_cam import BaseCAM
from pytorch_grad_cam.utils.find_layers import find_layer_predicate_recursive
from pytorch_grad_cam.utils.image import scale_accross_batch_and_channels, scale_cam_image
from pytorch_grad_cam.utils.svd_on_activations import get_2d_projection

class FullUnifiedCAM(BaseCAM):
    def __init__(self, model, target_layers, reshape_transform=None):
        if not isinstance(target_layers, list) or len(target_layers) <= 0:
            def layer_with_2D_bias(layer):
                bias_target_layers = [torch.nn.Conv2d, torch.nn.BatchNorm2d]
                if type(layer) in bias_target_layers and layer.bias is not None:
                    return True
                return False
            
            target_layers = find_layer_predicate_recursive(
                model, layer_with_2D_bias)
            
            print(f"INFO: {len(target_layers)} bias layers will be calculated.")

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

        cam_per_layer = self.compute_cam_per_layer(input_tensor=input_tensor)
        return self.aggregate_multi_layers(input_tensor=input_tensor, targets=targets, cam_per_target_layer=cam_per_layer, eigen_smooth=eigen_smooth)
        
    def compute_cam_per_layer(
        self, input_tensor: torch.Tensor
    ) -> torch.Tensor:
        activations_list = [a for a in self.activations_and_grads.activations]
        grads_list = [g for g in self.activations_and_grads.gradients]

        # Calculate feature map from input
        input_grad = input_tensor.grad.data.cpu().numpy()
        input_feature_map = input_grad * input_tensor.data.cpu().numpy()
        input_feature_map = np.abs(input_feature_map)
        input_feature_map = torch.from_numpy(scale_accross_batch_and_channels(input_feature_map, self.get_target_width_height(input_tensor))).to(self.device)

        # Loop over the saliency image from every layer
        cam_per_target_layer = input_feature_map
        for i in range(len(self.target_layers)):
            target_layer = self.target_layers[i]
            layer_activations = None
            layer_grads = None
            if i < len(activations_list):
                layer_activations = activations_list[i]
            if i < len(grads_list):
                layer_grads = grads_list[i]

            cam = self.get_cam_layer(input_tensor=input_tensor, target_layer=target_layer, activations=layer_activations, layer_grads=layer_grads)
            cam_per_target_layer = torch.cat([cam_per_target_layer, cam], dim=1)

        return cam_per_target_layer
    
    def get_cam_layer(
        self,
        input_tensor: torch.Tensor,
        target_layer: torch.nn.Module,
        activations: torch.Tensor,
        layer_grads: torch.Tensor,
    ) -> torch.Tensor:
        # Get the layer mask
        mask = self.get_layer_mask(activations=activations, layer_grads=layer_grads)

        with torch.no_grad():
            # Filter activations using the mask
            filtered_activations = torch.stack([
                activations[batch_idx][mask[batch_idx]] for batch_idx in range(activations.shape[0])
            ]).to(self.device)

            # Filter and normalize layer grads
            filtered_layer_grads = torch.stack([
                layer_grads[batch_idx][mask[batch_idx]] for batch_idx in range(layer_grads.shape[0])
            ]).to(self.device)
            eps = 1e-7
            sum_activations = filtered_activations.sum(dim=(2, 3), keepdim=True)
            pixel_weights = torch.nn.Softmax(dim=-1)(
                filtered_layer_grads * filtered_activations / (sum_activations + eps)
            )

            # Filter and normalize biases
            try:
                biases = self.get_bias_data(target_layer)
                filtered_biases = torch.stack([
                    biases[mask[batch_idx]] for batch_idx in range(mask.shape[0])
                ])[:, :, None, None]
            except:
                # If the layer doesn't have bias
                filtered_biases = torch.zeros_like(filtered_activations, device=self.device)

            # Highlight important pixels in each activation
            modified_activations = pixel_weights * filtered_activations

            # Calculate activation feature maps
            upsample = torch.nn.UpsamplingBilinear2d(size=input_tensor.shape[-2:])
            activation_feature_maps = upsample(modified_activations)

            # Calculate bias feature maps
            gradient_multiplied_biases = np.abs(filtered_biases.cpu().numpy() * filtered_layer_grads.cpu().numpy())
            bias_feature_maps = torch.from_numpy(scale_accross_batch_and_channels(gradient_multiplied_biases, self.get_target_width_height(input_tensor))).to(self.device)

            # Concantenate feature maps
            feature_maps = torch.cat([activation_feature_maps, bias_feature_maps], dim=1)
            feature_maps = self.normalize_feature_maps_minmax(feature_maps=feature_maps)

            return feature_maps
        
    def get_layer_mask(
        self,
        activations: torch.Tensor,
        layer_grads: torch.Tensor,
    ) -> torch.Tensor:
        # Calculate filtering weights using GradCAM++ approach
        grads_power_2 = layer_grads**2
        grads_power_3 = grads_power_2 * layer_grads
        # Equation 19 in https://arxiv.org/abs/1710.11063
        sum_activations = activations.sum(dim=(2, 3), keepdim=True)
        eps = 1e-6
        aij = grads_power_2 / (2 * grads_power_2 + sum_activations * grads_power_3 + eps)
        # Apply ReLU from Eq. 7 in the paper and zero out aij where activations are 0
        aij = torch.where(layer_grads != 0, aij, torch.zeros_like(aij))

        grad_cam_weights = torch.relu(layer_grads) * aij
        grad_cam_weights = grad_cam_weights.sum(dim=(2, 3))

        # Create the mask: keep weights greater than 0
        mask = grad_cam_weights > 0

        return mask

    def get_bias_data(self, layer: torch.nn.Module) -> torch.Tensor:
        if isinstance(layer, torch.nn.BatchNorm2d):
            bias = - (layer.running_mean * layer.weight / torch.sqrt(layer.running_var + layer.eps)) + layer.bias
            return bias
        else:
            return layer.bias
        
    def normalize_feature_maps_minmax(self, feature_maps: torch.Tensor) -> torch.Tensor:
        maxs = feature_maps.view(feature_maps.size(0),
                                feature_maps.size(1), -1).max(dim=-1)[0]
        mins = feature_maps.view(feature_maps.size(0),
                                feature_maps.size(1), -1).min(dim=-1)[0]

        maxs, mins = maxs[:, :, None, None], mins[:, :, None, None]
        feature_maps = (feature_maps - mins) / (maxs - mins + 1e-8)

        return feature_maps
    
    def aggregate_multi_layers(
        self, input_tensor: torch.Tensor,
        targets: List[torch.nn.Module],
        cam_per_target_layer: torch.Tensor,
        eigen_smooth: bool
    ) -> np.ndarray:
        # Pertubate input with feature maps
        pertubated_inputs = input_tensor[:, None, :, :] * cam_per_target_layer[:, :, None, :, :]

        if hasattr(self, "batch_size"):
            BATCH_SIZE = self.batch_size
        else:
            BATCH_SIZE = 16

        # Calculate score of each pertubated inputs
        scores = []
        for target, tensor in zip(targets, pertubated_inputs):
            for i in tqdm.tqdm(range(0, tensor.size(0), BATCH_SIZE)):
                batch = tensor[i: i + BATCH_SIZE, :]
                outputs = [target(o).to(self.device).item()
                            for o in self.model(batch)]
                scores.extend(outputs)
        scores = torch.Tensor(scores)
        scores = scores.view(pertubated_inputs.shape[0], pertubated_inputs.shape[1])
        weights = torch.nn.Softmax(dim=-1)(scores).numpy()

        # Aggregate activations to saliency map
        cam_per_target_layer = cam_per_target_layer.cpu().numpy()

        # 2D conv
        if len(cam_per_target_layer.shape) == 4:
            weighted_activations = weights[:, :, None, None] * cam_per_target_layer
        # 3D conv
        elif len(cam_per_target_layer.shape) == 5:
            weighted_activations = weights[:, :, None, None, None] * cam_per_target_layer
        else:
            raise ValueError(f"Invalid activation shape. Get {len(cam_per_target_layer.shape)}.")
        
        # ReLU activation values
        weighted_activations = np.maximum(weighted_activations, 0)

        if eigen_smooth:
            cam = get_2d_projection(weighted_activations)
        else:
            cam = weighted_activations.sum(axis=1)

        return scale_cam_image(cam)