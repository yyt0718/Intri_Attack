import torch
import torchvision.models as models

def batch_norm_as_linear_transformation_sparse(X, gamma, beta, moving_mean, moving_var, eps=1e-5):
    """
    Implement batch normalization as a linear transformation using sparse matrices.

    Parameters:
    X (Tensor): Input data of shape (C, H, W)
    gamma (Tensor): Scale factor, of shape (C,)
    beta (Tensor): Shift factor, of shape (C,)
    moving_mean (Tensor): Moving mean, of shape (C,)
    moving_var (Tensor): Moving variance, of shape (C,)
    eps (float): A small number to avoid division by zero

    Returns:
    Tensor: Batch-normalized data Y
    """
    C, H, W = X.shape
    X_flat = X.reshape(-1, 1)  # Fully flatten X to (C*H*W, 1)

    # Create the scaled diagonal matrix A
    gamma_scaled = gamma / torch.sqrt(moving_var + eps)
    gamma_scaled_repeated = gamma_scaled.repeat_interleave(H * W)
    indices = torch.arange(0, len(gamma_scaled_repeated)).repeat(2, 1)
    values = gamma_scaled_repeated
    i = torch.tensor(indices, dtype=torch.long)
    v = torch.tensor(values, dtype=torch.float32)
    A_sparse = torch.sparse_coo_tensor(i, v, (len(gamma_scaled_repeated), len(gamma_scaled_repeated)))

    # Create the shifted vector B
    beta_shifted = beta - moving_mean * gamma / torch.sqrt(moving_var + eps)
    B = beta_shifted.repeat_interleave(H * W).unsqueeze(1)

    # Apply the sparse matrix transformation Y = AX + B
    Y = torch.sparse.mm(A_sparse, X_flat) + B

    # Reshape Y back to the original shape of X
    Y = Y.reshape(C, H, W)

    return Y


# Example usage:
# gamma, beta, moving_mean, moving_var = [obtained from a BatchNorm layer or defined]
# X = torch.randn(C, H, W)  # Example input tensor
# Y = batch_norm_as_linear_transformation(X, gamma, beta, moving_mean, moving_var)








# Load a pretrained ResNet-18 model
resnet152 = models.resnet152(pretrained=True).eval()

# Select a BatchNorm2d layer from the ResNet-18 model
# For this example, we will take the first BatchNorm2d layer
bn_layer = resnet152.layer2[0].bn3

# Check if a BatchNorm2d layer was found
if bn_layer is not None:
    # Example input tensor without batch dimension
    X_simple = torch.randn(512, 28,28)  # Random data: 5 channels, 5x5 pixels each

    # Apply the ResNet's BatchNorm2d layer
    resnet_bn_output = bn_layer(X_simple.unsqueeze(0)).squeeze(0)

    # Apply the custom fully flattened batch normalization
    custom_bn_output = batch_norm_as_linear_transformation_sparse(X_simple, bn_layer.weight.data, bn_layer.bias.data, bn_layer.running_mean.data, bn_layer.running_var.data)

    # Compare the outputs
    output_difference_resnet = torch.abs(resnet_bn_output - custom_bn_output)
    mean_difference_resnet = output_difference_resnet.mean().item()
    max_difference_resnet = output_difference_resnet.max().item()

    print("Mean difference:", mean_difference_resnet)
    print("Max difference:", max_difference_resnet)
    print(output_difference_resnet)
else:
    print("No BatchNorm2d layer found in ResNet-18 model.")