def replace_layer_recursive(model, old_layer, new_layer):
    for name, layer in model._modules.items():
        if layer == old_layer:
            model._modules[name] = new_layer
            return True
        elif replace_layer_recursive(layer, old_layer, new_layer):
            return True
    return False


def replace_all_layer_type_recursive(model, old_layer_type, new_layer):
    for name, layer in model._modules.items():
        if isinstance(layer, old_layer_type):
            model._modules[name] = new_layer
        replace_all_layer_type_recursive(layer, old_layer_type, new_layer)


def find_layer_types_recursive(model, layer_types):
    def predicate(layer):
        return type(layer) in layer_types
    return find_layer_predicate_recursive(model, predicate)


def find_layer_predicate_recursive(model, predicate):
    result = []
    for name, layer in model._modules.items():
        if predicate(layer):
            result.append(layer)
        result.extend(find_layer_predicate_recursive(layer, predicate))
    return result

def find_layer(model, target_layer_name):
    """Find target layer to calculate CAM.

        : Args:
            - **model - **: Self-defined model architecture.
            - **target_layer_name - ** (str): Name of target class.

        : Return:
            - **target_layer - **: Found layer. This layer will be hooked to get forward/backward pass information.
    """
    
    target_layer = model
    layer_names = target_layer_name.split('.')

    for name in layer_names:
        if isinstance(target_layer, list):
            try:
                name = int(name)
            except ValueError:
                raise Exception(f"Expected an index in list but got {name} instead.")
            
            if name >= len(target_layer):
                raise IndexError(f"Index {name} out of range for layer list.")
            
            target_layer = target_layer[name]
        else:
            if not hasattr(target_layer, name):
                raise Exception(f"Invalid target layer name: {target_layer_name}")
            target_layer = getattr(target_layer, name)
    
    return target_layer
