import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def num_parameters(model):
    """
    Given a model, return the total number of parameters.
    :param model:
    :return: total_parms, trainable_params
    """

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

