import random

import numpy as np
import torch
import torch.nn.functional as F


def add_noise(data):
    noise = torch.randn_like(data) * 0.1
    return data + noise


iteration_counter = 0


def train_dis(iter_idx, dis_optim, pose_decoder, time_seq, in_text, in_audio, vid_indices, discriminator, target_poses,
              ret_dict, args, audio_var_bottom, audio_var_top):
    dis_error = None
    dis_optim.zero_grad()

    pre_seq = target_poses.new_zeros((target_poses.shape[0], target_poses.shape[1], target_poses.shape[2] + 1))
    pre_seq[:, 0:args.n_pre_poses, :-1] = target_poses[:, 0:args.n_pre_poses]
    pre_seq[:, 0:args.n_pre_poses, -1] = 1  # indicating bit for constraints

    beat_out, *_ = pose_decoder(pre_seq, time_seq, in_text, in_audio, audio_var_bottom, audio_var_top
                                , vid_indices=vid_indices)  # out shape (batch x seq x dim)

    dis_real = discriminator(target_poses, in_text, in_audio, audio_var_bottom, audio_var_top)
    dis_fake_beat = discriminator(beat_out, in_text, in_audio, audio_var_bottom, audio_var_top)

    # dis_error = torch.sum(-torch.mean(torch.log(dis_real + 1e-8) + torch.log(1 - dis_fake + 1e-8)))  # ns-gan
    # beat_error = torch.sum(-torch.mean(torch.log(dis_real + 1e-8) + torch.log(1 - dis_fake_beat + 1e-8)))  # ns-gan
    beat_error = -torch.mean(dis_real) + torch.mean(dis_fake_beat)
    # all_error = dis_error + beat_error
    g_div = calculate_gradient_penalty(target_poses, beat_out, dis_real, dis_fake_beat)
    ret_dict['gradientPenalty_discriminator'] = g_div
    all_error = beat_error + g_div
    ret_dict['dis'] = all_error.sum().cpu().detach().numpy()

    all_error.backward()
    dis_optim.step()

    return ret_dict


def train_vae(iter_idx, vae_optim, vae, target_poses, ret_dict):
    l_out = None
    num = 16
    for i in random.sample(range(0, 34), num):
        vae_optim.zero_grad()
        dec_pose, diff_pose, quant_t_pose, quant_b_pose = vae(target_poses[:, i, :])
        pre_recon_loss1 = F.smooth_l1_loss(dec_pose, target_poses[:, i, :])
        pre_latent_loss = diff_pose.mean()

        vq_vae_loss1 = pre_recon_loss1 + 0.05 * pre_latent_loss
        l_out = vq_vae_loss1 if l_out is None else l_out + vq_vae_loss1

        vq_vae_loss1.backward()
        vae_optim.step()

    l_out = l_out / num

    ret_dict["pose_vae_loss"] = l_out.cpu().detach().numpy()

    return ret_dict


def velocity_loss(target_poses, out_beat):
    beta = 0.1
    out = None
    for i in range(1, target_poses.shape[1]):
        l1 = target_poses[:, i, :] - target_poses[:, i - 1, :]
        l2 = out_beat[:, i, :] - out_beat[:, i - 1, :]
        temp = F.smooth_l1_loss(l2 / beta, l1 / beta) * beta
        out = out + temp if out is not None else temp
    return out


def consistency_loss(out_beat):
    beta = 0.1
    out = None
    for i in range(1, out_beat.shape[1]):
        temp = F.smooth_l1_loss(out_beat[:, i, :] / beta, out_beat[:, i - 1, :] / beta) * beta
        out = out + temp if out is not None else temp
    return out


def p1_loss(args, target_poses, out_beat):
    beta = 0.1
    t1 = target_poses[:, :args.n_pre_poses, :]
    t2 = out_beat[:, :args.n_pre_poses, :]
    return F.smooth_l1_loss(t2 / beta, t1 / beta) * beta


def train_gen(iter_idx, pose_dec_optim, pose_decoder, target_poses, args, time_seq, in_text, in_audio, vid_indices,
              discriminator, jerkLoss, ret_dict, epoch, audio_var_bottom, audio_var_top, warm_up_epochs,
              train_gan=False):
    pose_dec_optim.zero_grad()
    pre_seq = target_poses.new_zeros((target_poses.shape[0], target_poses.shape[1], target_poses.shape[2] + 1))
    pre_seq[:, 0:args.n_pre_poses, :-1] = target_poses[:, 0:args.n_pre_poses]
    pre_seq[:, 0:args.n_pre_poses, -1] = 1  # indicating bit for constraints

    # decoding
    out_beat, z, z_mu, z_logvar, vqvae_loss = pose_decoder(pre_seq, time_seq, in_text, in_audio, audio_var_bottom,
                                                           audio_var_top, vid_indices)

    # loss
    beta = 0.1

    out_base = out_beat.reshape((out_beat.shape[0], out_beat.shape[1], 53, 3))
    target_base = target_poses.reshape((target_poses.shape[0], target_poses.shape[1], 53, 3))

    out_body = out_base[:, :, :11, :]
    target_body = target_base[:, :, :11, :]

    out_left_hand = out_base[:, :, 11:11 + 21, :]
    target_left_hand = target_base[:, :, 11:11 + 21, :]

    out_right_hand = out_base[:, :, 11:11 + 42, :]
    target_right_hand = target_base[:, :, 11:11 + 42, :]

    huber_loss_body = F.smooth_l1_loss(out_body / beta, target_body / beta) * beta
    huber_loss_left_hand = F.smooth_l1_loss(out_left_hand / beta, target_left_hand / beta) * beta
    huber_loss_right_hand = F.smooth_l1_loss(out_right_hand / beta, target_right_hand / beta) * beta

    ret_dict["huber_body_loss"] = huber_loss_body
    ret_dict["huber_left_hand_loss"] = huber_loss_left_hand
    ret_dict["huber_right_hand_loss"] = huber_loss_right_hand

    beat_huber_loss = (huber_loss_body + huber_loss_left_hand + huber_loss_right_hand)

    delta_target = target_poses[:, 1:, :] - target_poses[:, :-1, :]
    delta_beat = out_beat[:, 1:, :] - out_beat[:, :-1, :]
    delta_beat_huber_loss = F.smooth_l1_loss(delta_beat / beta, delta_target / beta) * beta

    if epoch >= warm_up_epochs and train_gan:
        beat_output = discriminator(out_beat, in_text, in_audio, audio_var_bottom, audio_var_top)
        beat_error = -torch.mean(beat_output)
    kld = div_reg = None

    beat_div_reg = None
    rand_idx = torch.randperm(vid_indices.shape[0])
    rand_vids = vid_indices[rand_idx]
    out_beat_vec_rand_vid, z_rand_vid, _, _, _ = pose_decoder(pre_seq, time_seq, in_text, in_audio, audio_var_bottom,
                                                              audio_var_top, rand_vids)
    beta = 0.05
    beat_pose_l1 = F.smooth_l1_loss(out_beat / beta, out_beat_vec_rand_vid.detach() / beta, reduction='none') * beta
    beat_pose_l1 = beat_pose_l1.sum(dim=1).sum(dim=1)
    beat_pose_l1 = beat_pose_l1.view(beat_pose_l1.shape[0], -1).mean(1)

    z_l1 = F.l1_loss(z.detach(), z_rand_vid.detach(), reduction='none')
    z_l1 = z_l1.view(z_l1.shape[0], -1).mean(1)
    beat_div_reg = -(beat_pose_l1 / (z_l1 + 1.0e-5))
    # beat_div_reg = -(beat_pose_l1 )
    beat_div_reg = torch.clamp(beat_div_reg, min=-1000)
    beat_div_reg = beat_div_reg.mean()

    j1, j2, j3 = jerkLoss(out_beat, target_poses)

    position_loss = p1_loss(args, target_poses, out_beat)
    # con_loss = consistency_loss(out_beat)

    ret_dict['start_position_error'] = position_loss
    ret_dict['velocity_loss'] = j1
    ret_dict['acceleration_loss'] = j2
    ret_dict['jerk_loss'] = j3
    ret_dict['vqvae_loss'] = vqvae_loss
    # ret_dict['consistency_loss'] = con_loss

    # speaker embedding KLD
    kld = -0.5 * torch.mean(1 + z_logvar - z_mu.pow(2) - z_logvar.exp())
    # loss = args.loss_regression_weight * (delta_huber_loss + delta_beat_huber_loss) + args.loss_regression_weight * (huber_loss + beat_huber_loss) + args.loss_kld_weight * kld + args.loss_reg_weight * (div_reg + beat_div_reg)
    loss = args.loss_regression_weight * (j1 + j2 + j3)
    loss += args.loss_regression_weight * beat_huber_loss
    loss += args.loss_regression_weight * delta_beat_huber_loss
    loss += args.loss_regression_weight * vqvae_loss

    loss += args.loss_kld_weight * kld

    # loss += args.loss_reg_weight * con_loss

    # loss += args.loss_gan_weight/2 * jerk_loss
    # loss += args.loss_gan_weight/2 * position_loss

    if epoch >= warm_up_epochs and train_gan:
        loss += args.loss_gan_weight * beat_error
        ret_dict['gen'] = beat_error.sum()

    # pdb.set_trace()

    ret_dict['beat_hubert_loss'] = beat_huber_loss.sum()
    ret_dict['delta_beat_hubert_loss'] = delta_beat_huber_loss.sum()
    if kld:
        ret_dict['KLD'] = kld.sum()
    if beat_div_reg:
        ret_dict['DIV_REG'] = beat_div_reg.sum()
        loss += args.loss_reg_weight * beat_div_reg

    loss.backward()
    pose_dec_optim.step()

    return ret_dict


def makeVaeLatent(vae, target_poses):
    ret = target_poses.new_zeros((target_poses.shape[0], target_poses.shape[1], 320))
    for i in range(target_poses.shape[1]):
        vv = vae.encode(target_poses[:, i, :])
        ret[:, i, :] = vv
    return ret


def train_beat_gan(iter_idx, args, epoch, time_seq, in_text, in_audio, audio_var_bottom, audio_var_top, target_poses,
                   vid_indices,
                   pose_decoder, discriminator,
                   pose_dec_optim, dis_optim, velLoss):
    global iteration_counter
    warm_up_epochs = args.loss_warmup
    use_noisy_target = False

    target_disc = torch.autograd.Variable(target_poses.type(torch.Tensor), requires_grad=True).cuda()

    # make pre seq input
    iteration_counter += 1

    ret_dict = {}

    ###########################################################################################
    # train D

    # ret_dict = train_vae(iter_idx, vae_optim,vae, target_poses,ret_dict)

    ret_dict = train_dis(iter_idx, dis_optim, pose_decoder, time_seq, in_text, in_audio, vid_indices, discriminator,
                         target_disc, ret_dict, args, audio_var_bottom, audio_var_top)
    train_gan = iter_idx % 3 == 0 and float(ret_dict["dis"]) < 20
    ret_dict["using_gan"] = 1 if train_gan else 0
    ret_dict = train_gen(iter_idx, pose_dec_optim, pose_decoder, target_poses, args, time_seq, in_text, in_audio,
                         vid_indices, discriminator, velLoss, ret_dict, epoch, audio_var_bottom, audio_var_top,
                         warm_up_epochs, train_gan=train_gan)

    return ret_dict


from torch import Tensor
import torch.autograd as autograd
from torch.autograd import Variable


def compute_gradient_penalty(D, real_samples, fake_samples, in_text, in_audio):
    """Calculates the gradient penalty loss for WGAN GP"""
    # real_samples = real_samples.reshape((real_samples.shape[0],48*3,1,1))
    # fake_samples = fake_samples.reshape((fake_samples.shape[0],48*3,1,1))
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random(real_samples.shape)).cuda()
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates, in_text, in_audio)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False).cuda()
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def calculate_gradient_penalty(real_data, fake_data, real_outputs, fake_outputs, k=2, p=6, device=torch.device("cuda")):
    real_grad_outputs = torch.full((real_data.size(0), 1), 1, dtype=torch.float32, requires_grad=False, device=device)
    fake_grad_outputs = torch.full((fake_data.size(0), 1), 1, dtype=torch.float32, requires_grad=False, device=device)
    # real_data = real_data.reshape((real_data.shape[0], -1))
    # fake_data = fake_data.reshape((fake_data.shape[0], -1))

    real_gradient = torch.autograd.grad(
        outputs=real_outputs,
        inputs=real_data,
        grad_outputs=real_grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    fake_gradient = torch.autograd.grad(
        outputs=fake_outputs,
        inputs=fake_data,
        grad_outputs=fake_grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    real_gradient_norm = real_gradient.reshape(real_gradient.size(0), -1).pow(2).sum(1) ** (p / 2)
    fake_gradient_norm = fake_gradient.reshape(fake_gradient.size(0), -1).pow(2).sum(1) ** (p / 2)

    gradient_penalty = torch.mean(real_gradient_norm + fake_gradient_norm) * k / 2
    return gradient_penalty
