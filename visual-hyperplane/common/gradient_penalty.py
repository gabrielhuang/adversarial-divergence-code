import torch


def compute_gradient_penalty(netD, data):
    # Dunnow how to do this better with detach
    data = torch.autograd.Variable(data.detach(), requires_grad=True)
    outputs = netD(data)

    gradients = torch.autograd.grad(outputs=outputs,
                                    inputs=data,
                                    grad_outputs=torch.ones(outputs.size()),
                                    create_graph=True,
                                    retain_graph=True,
                                    only_inputs=True)[0]

    # Careful with the dimensions!! The input is multidimensional
    #old_gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    #new_gradient_penalty = torch.max((gradients**2).sum() - 1., torch.zeros(1))
    relaxed_gradient_penalty = (gradients**2).sum() / float(len(data))
    return relaxed_gradient_penalty


def compute_gradient_penalty_logits(netD, data):
    # Dunnow how to do this better with detach
    data = torch.autograd.Variable(data.detach(), requires_grad=True)
    outputs = netD.get_logits(data)

    gradients = torch.autograd.grad(outputs=outputs,
                                    inputs=data,
                                    grad_outputs=torch.ones(outputs.size()),
                                    create_graph=True,
                                    retain_graph=True,
                                    only_inputs=True)[0]

    # Careful with the dimensions!! The input is multidimensional
    #old_gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    #new_gradient_penalty = torch.max((gradients**2).sum() - 1., torch.zeros(1))
    relaxed_gradient_penalty = (gradients**2).sum() / float(len(data))
    return relaxed_gradient_penalty
