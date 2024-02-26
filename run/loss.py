import torch
from torch.autograd import Variable
from torch.autograd import grad as torch_grad


def calc_gradient_penalty(netD, real_data, generated_data):
    # GP strength
    LAMBDA = 0.1

    b_size = real_data.size()[0]

    # Calculate interpolation
    alpha = torch.rand(b_size, 1, 1, 1)
    alpha = alpha.expand_as(real_data)
    alpha = alpha.cuda()

    interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
    interpolated = Variable(interpolated, requires_grad=True)
    interpolated = interpolated.cuda()

    # Calculate probability of interpolated examples
    prob_interpolated = netD(interpolated)

    # Calculate gradients of probabilities with respect to examples
    gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                           grad_outputs=torch.ones(prob_interpolated.size()).cuda(),
                           create_graph=True, retain_graph=True)[0]

    # Gradients have shape (batch_size, num_channels, img_width, img_height),
    # so flatten to easily take norm per example in batch
    gradients = gradients.view(b_size, -1)

    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

    # Return gradient penalty
    return LAMBDA * (((gradients_norm - 100) ** 2) / 100 ** 2).mean()


def get_freq(data):
    freq = torch.fft.fft2(data, norm='ortho')
    freq = torch.stack([freq.real, freq.imag], -1)

    return freq


def fourier_transform_loss(criterion_fft, fourier_transform_weight, real_A, fake_A, cycle_A, real_B, fake_B, cycle_B):
    A_label_fft = get_freq(real_A)
    A_fake_fft = get_freq(fake_A)
    A_cycle_fft = get_freq(cycle_A)

    B_label_fft = get_freq(real_B)
    B_fake_fft = get_freq(fake_B)
    B_cycle_fft = get_freq(cycle_B)

    loss_hf_A = criterion_fft(B_fake_fft, A_label_fft)
    loss_hf_B = criterion_fft(A_fake_fft, B_label_fft)

    loss_hf_cycle_A = criterion_fft(A_fake_fft, A_cycle_fft)
    loss_hf_cycle_B = criterion_fft(B_fake_fft, B_cycle_fft)

    fourier_loss = fourier_transform_weight * (loss_hf_A + loss_hf_B + loss_hf_cycle_A + loss_hf_cycle_B)

    return fourier_loss