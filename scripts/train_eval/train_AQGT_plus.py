import numpy as np
import torch
import torch.nn.functional as F


iteration_counter = 0

def train_dis(generator, beat_seeg, in_text, in_audio, vid_indices, discriminator, target_poses, ret_dict, args, audio_var_bottom, audio_var_top, anno):
    """
    Training function for the discriminator
    @param generator: the generator model
    @param beat_seeg: Seeg audio information
    @param in_text: text input
    @param in_audio: audio input
    @param vid_indices: the speaker identity
    @param discriminator: the discriminator model
    @param target_poses: the ground truth gestures
    @param ret_dict: the return dictionary for loss reporting
    @param args: config arg parse file
    @param audio_var_bottom: audio vqvae bottom vector
    @param audio_var_top: audio vqvae top vector
    @param anno: the annotation file
    @return: ret_dict for loss reporting and the final loss
    """
    target_poses = Variable(target_poses.type(Tensor), requires_grad=True).cuda()

    pre_seq = target_poses.new_zeros((target_poses.shape[0], target_poses.shape[1], target_poses.shape[2] + 1))
    pre_seq[:, 0:args.n_pre_poses, :-1] = target_poses[:, 0:args.n_pre_poses]
    pre_seq[:, 0:args.n_pre_poses, -1] = 1  # indicating bit for constraints

    beat_out, *_ = generator(pre_seq, beat_seeg, in_text, in_audio, audio_var_bottom, audio_var_top,
                             vid_indices, anno)  # out shape (batch x seq x dim)

    dis_real = discriminator(target_poses, in_text, in_audio, audio_var_bottom, audio_var_top,anno)
    dis_fake_beat = discriminator(beat_out, in_text, in_audio, audio_var_bottom, audio_var_top,anno)

    beat_error = -torch.mean(dis_real) + torch.mean(dis_fake_beat)
    g_div = calculate_gradient_penalty(target_poses, beat_out, dis_real, dis_fake_beat)
    ret_dict['gradientPenalty_discriminator'] = g_div
    all_error = beat_error + g_div
    ret_dict['dis'] = all_error.sum()

    return ret_dict,all_error


def p1_loss(args,target_poses,out_beat):
    """
    Simple loss between the ground truth and the generated gestures.
    @param args: config arg
    @param target_poses: ground truth poses
    @param out_beat: generated gestures
    @return:
    """
    beta = 0.1
    t1 = target_poses[:,:args.n_pre_poses,:]
    t2 = out_beat[:,:args.n_pre_poses,:]
    return F.smooth_l1_loss(t2 / beta, t1 / beta) * beta


def reboneUpper(d3_pose):
    """
    the original data is a unnormalized direction vector. This function changes the direction vector to be a positional vector in world space.
    @param d3_pose: direction gesture poses in
    @return: world space gestures out
    """
    ubs = 6
    skeleton_parents = np.asarray([-1, 0, 7 - ubs, 8 - ubs, 9 - ubs, 8 - ubs, 11 - ubs, 12 - ubs, 8 - ubs, 14 - ubs, 15 - ubs])
    hand_parents_l = np.asarray([-4, -4, 1, 2, 3, -4, 5, 6, 7, -4, 9, 10, 11, -4, 13, 14, 15, -4, 17, 18, 19])
    hand_parents_r = np.asarray([-22, -22, 1, 2, 3, -22, 5, 6, 7, -22, 9, 10, 11, -22, 13, 14, 15, -22, 17, 18, 19])
    hand_parents_l = hand_parents_l + 17 - ubs
    hand_parents_r = hand_parents_r + 17 + 21 - ubs
    skeleton_parents = np.concatenate((skeleton_parents, hand_parents_l, hand_parents_r), axis=0)
    out = []
    for i in range(53):
        parent = skeleton_parents[i]
        if parent != -1:
            parent_bone = d3_pose[:,:,parent,:]
            new_bone = d3_pose[:,:,i,:] + parent_bone
            d3_pose[:,:,i,:] = new_bone
            out.append(new_bone.unsqueeze(2))
        else:
            out.append(d3_pose[:,:,i,:].unsqueeze(2))
    return torch.cat(out,dim=2)
def convert_dir_vec_to_pose(vec):
    return reboneUpper(vec)

def calculate_distance(a,b):
    """
    Simple L1 distance
    @param a: vector a
    @param b: vector b
    @return:
    """
    return (a.reshape((a.shape[0]*a.shape[1],-1)) - b.reshape((b.shape[0]*b.shape[1],-1)))

def calc_distance_loss(target_poses,output_gestures):
    """
    additional loss functions
    @param target_poses: the ground truth gestures
    @param output_gestures: the generated gestures
    @return: loss information
    """
    beta = 0.1
    target_poses = convert_dir_vec_to_pose(target_poses.clone().reshape((-1,34,53,3)))
    output_gestures = convert_dir_vec_to_pose(output_gestures.clone().reshape((-1,34,53,3)))

    # loss function for the gestures in recreated position space
    pose_loss = F.smooth_l1_loss(output_gestures / beta, target_poses / beta) * beta

    elbow_out = calculate_distance(output_gestures[:, :, 6, :], output_gestures[:, :, 9, :])
    hand_out = calculate_distance(output_gestures[:, :, 7, :], output_gestures[:, :, 10, :])
    elbow_target = calculate_distance(target_poses[:, :, 6, :], target_poses[:, :, 9, :])
    hand_target = calculate_distance(target_poses[:, :, 7, :], target_poses[:, :, 10, :])

    # position loss for the lower arm and hand positions.
    position_loss = F.smooth_l1_loss(elbow_out / beta, elbow_target / beta) * beta
    position_loss += F.smooth_l1_loss(hand_out / beta, hand_target / beta) * beta
    position_loss /= 2

    ubs = 6
    skeleton_parents = np.asarray([-1, 0, 7 - ubs, 8 - ubs, 9 - ubs, 8 - ubs, 11 - ubs, 12 - ubs, 8 - ubs, 14 - ubs, 15 - ubs])
    hand_parents_l = np.asarray([-4, -4, 1, 2, 3, -4, 5, 6, 7, -4, 9, 10, 11, -4, 13, 14, 15, -4, 17, 18, 19])
    hand_parents_r = np.asarray([-22, -22, 1, 2, 3, -22, 5, 6, 7, -22, 9, 10, 11, -22, 13, 14, 15, -22, 17, 18, 19])
    hand_parents_l = hand_parents_l + 17 - ubs
    hand_parents_r = hand_parents_r + 17 + 21 - ubs

    skeleton_parents = np.concatenate((skeleton_parents, hand_parents_l, hand_parents_r), axis=0)

    # bone_length_loss
    bone_loss = None
    for i in range(1,53):
        p_d_out = calculate_distance(output_gestures[:,:,i,:],output_gestures[:,:,skeleton_parents[i],:])
        p_d_target = calculate_distance(target_poses[:, :, i, :], target_poses[:, :, skeleton_parents[i], :])
        l1 = F.smooth_l1_loss(p_d_out / beta, p_d_target / beta) * beta
        bone_loss = l1 if bone_loss is None else bone_loss + l1
    bone_loss /= 52

    # finger_distance_loss
    distance_loss = None
    for i in range(0,21):
        p_d_out = calculate_distance(output_gestures[:,:,12+i,:],output_gestures[:,:,12+20+i,:])
        p_d_target = calculate_distance(target_poses[:, :, 12+i, :], target_poses[:, :, 12+20+i, :])
        l1 = F.smooth_l1_loss(p_d_out / beta, p_d_target / beta) * beta
        distance_loss = l1 if distance_loss is None else distance_loss + l1
    distance_loss /= 21

    return pose_loss,position_loss,bone_loss,distance_loss









def train_gen(generator_model, target_poses, args, beat, in_text, in_audio, vid_indices, discriminator, jerkLoss, ret_dict, epoch, audio_var_bottom, audio_var_top, warm_up_epochs, anno, train_gan=False):
    """
    The training function for our generator
    @param generator_model: the generator model
    @param target_poses: the ground truth gestures
    @param args: a config arg file
    @param beat: the SEEG beat vector
    @param in_text: input text
    @param in_audio: input audio
    @param vid_indices: input speaker identity
    @param discriminator: The discriminator model
    @param jerkLoss: We explicitly set the jerk loss for legacy reasons. TODO: remove
    @param ret_dict: the return dict, that is used to collect information
    @param epoch: the current epoch
    @param audio_var_bottom: the audio vqvae bottom output
    @param audio_var_top:  the audio vqvae top output
    @param warm_up_epochs: number of warmup epchs. Currently not used
    @param anno: the input annotation
    @param train_gan: True if the gan loss should be calculated and appended
    @return: the ret dict with loss information and the final loss
    """
    pre_seq = target_poses.new_zeros((target_poses.shape[0], target_poses.shape[1], target_poses.shape[2] + 1))
    pre_seq[:, 0:args.n_pre_poses, :-1] = target_poses[:, 0:args.n_pre_poses]
    pre_seq[:, 0:args.n_pre_poses, -1] = 1  # indicating bit for constraints

    # decoding
    out_beat, z, z_mu, z_logvar, arc_loss, ret_anno = generator_model(pre_seq, beat, in_text, in_audio, audio_var_bottom, audio_var_top,
                                                                      vid_indices, anno)
    for k,v in ret_anno.items():
        ret_dict["anno_"+k] = v

    # loss
    beta = 0.1

    out_base = out_beat.reshape((out_beat.shape[0], out_beat.shape[1], 53, 3))
    target_base = target_poses.reshape((target_poses.shape[0], target_poses.shape[1], 53, 3))

    out_body = out_base[:, :, :11, :]
    target_body = target_base[:, :, :11, :]

    out_left_hand = out_base[:, :, 11:11 + 21, :]
    target_left_hand = target_base[:, :, 11:11 + 21, :]

    out_right_hand = out_base[:, :, 11 + 21:11 + 42, :]
    target_right_hand = target_base[:, :, 11 + 21:11 + 42, :]

    huber_loss_body = F.smooth_l1_loss(out_body / beta, target_body / beta) * beta
    huber_loss_left_hand = F.smooth_l1_loss(out_left_hand / beta, target_left_hand / beta) * beta
    huber_loss_right_hand = F.smooth_l1_loss(out_right_hand / beta, target_right_hand / beta) * beta

    ret_dict["huber_body_loss"] = huber_loss_body
    ret_dict["huber_left_hand_loss"] = huber_loss_left_hand
    ret_dict["huber_right_hand_loss"] = huber_loss_right_hand

    beat_huber_loss = (huber_loss_body + (huber_loss_left_hand + huber_loss_right_hand)*4)

    delta_target = target_poses[:, 1:, :] - target_poses[:, :-1, :]
    delta_beat = out_beat[:, 1:, :] - out_beat[:, :-1, :]
    delta_beat_huber_loss = F.smooth_l1_loss(delta_beat / beta, delta_target / beta) * beta

    if train_gan:
        beat_output = discriminator(out_beat, in_text, in_audio, audio_var_bottom, audio_var_top,anno)
        beat_error = -torch.mean(beat_output)

    rand_idx = torch.randperm(vid_indices.shape[0])
    rand_vids = vid_indices[rand_idx]
    out_beat_vec_rand_vid, z_rand_vid, z_rand_mu, z_rand_logvar, arc_rand_loss, *_ = generator_model(pre_seq, beat, in_text, in_audio, audio_var_bottom, audio_var_top, rand_vids, anno)


    beat_pose_l1 = F.smooth_l1_loss(out_beat / beta, out_beat_vec_rand_vid.detach() / beta, reduction='none') * beta
    beat_pose_l1 = beat_pose_l1.sum(dim=1).sum(dim=1)
    beat_pose_l1 = beat_pose_l1.view(beat_pose_l1.shape[0], -1).mean(1)

    z_l1 = F.l1_loss(z.detach(), z_rand_vid.detach(), reduction='none')
    z_l1 = z_l1.view(z_l1.shape[0], -1).mean(1)
    beat_div_reg = -(beat_pose_l1 / (z_l1 + 1.0e-5))
    # beat_div_reg = -(beat_pose_l1 )
    beat_div_reg = torch.clamp(beat_div_reg, min=-1000)
    beat_div_reg = beat_div_reg.mean()

    position_loss = p1_loss(args, target_poses, out_beat)

    ret_dict['start_position_error'] = position_loss

    j1, j2, j3, j4, j5, j6, j7 = jerkLoss(out_beat, target_poses)

    ret_dict['velocity_loss'] = j1
    ret_dict['acceleration_loss'] = j2
    ret_dict['jerk_loss'] = j3
    ret_dict['jounce_loss'] = j4
    ret_dict['crackle_loss'] = j5
    ret_dict['pop_loss'] = j6
    ret_dict['lock_loss'] = j7
    ret_dict['arc_loss'] = arc_loss
    ret_dict['arc_rand_loss'] = arc_rand_loss

    pose_loss, position_loss, bone_loss, distance_loss = calc_distance_loss(target_poses,out_beat)
    ret_dict['pose_loss'] = pose_loss
    ret_dict['position_loss'] = position_loss
    ret_dict['bone_loss'] = bone_loss
    ret_dict['distance_loss'] = distance_loss

    # speaker embedding KLD
    kld = -0.5 * torch.mean(1 + z_logvar - z_mu.pow(2) - z_logvar.exp())
    loss = args.loss_regression_weight * (j1 + j2 + j3 + j4 + j5 + j6 + j7)
    loss += args.loss_kld_weight * pose_loss
    loss += args.loss_kld_weight * position_loss
    loss += args.loss_kld_weight * bone_loss
    loss += args.loss_kld_weight * distance_loss

    ret_dict['weighted_pose_loss'] = args.loss_kld_weight * pose_loss
    ret_dict['weighted_position_loss'] = args.loss_kld_weight * position_loss
    ret_dict['weighted_bone_loss'] = args.loss_kld_weight * bone_loss
    ret_dict['weighted_distance_loss'] = args.loss_kld_weight * distance_loss
    ret_dict["weighted_vaj"] = args.loss_regression_weight * (j1 + j2 + j3 + j4 + j5 + j6 + j7)
    loss += args.loss_regression_weight * beat_huber_loss
    ret_dict["weighted_beat_huber"] = args.loss_regression_weight * beat_huber_loss
    loss += args.loss_regression_weight * delta_beat_huber_loss
    ret_dict["weighted_delta_beat"] = args.loss_regression_weight * delta_beat_huber_loss

    loss += args.loss_kld_weight * kld
    ret_dict["weighted_kld"] = args.loss_kld_weight * kld
    loss += args.loss_vel_weight * arc_loss
    ret_dict["weighted_arc"] = args.loss_vel_weight * arc_loss

    ret_dict['beat_hubert_loss'] = beat_huber_loss.sum()
    ret_dict['delta_beat_hubert_loss'] = delta_beat_huber_loss.sum()
    if kld:
        ret_dict['KLD'] = kld.sum()
    if beat_div_reg:
        ret_dict['DIV_REG'] = beat_div_reg.sum()

        loss += args.loss_reg_weight * beat_div_reg
        ret_dict["weighted_beat_div"] = args.loss_reg_weight * beat_div_reg
        #loss += (-(args.loss_vel_weight * torch.clamp(arc_rand_loss,min=-1000))) / 1000
        #ret_dict["weighted_arc_rand"] = (-(args.loss_vel_weight * torch.clamp(arc_rand_loss,min=-1000))) / 1000

    ret_dict['loss'] = loss

    if train_gan:
        loss += args.loss_gan_weight * beat_error
        ret_dict["weighted_gan"] = args.loss_gan_weight * beat_error
        ret_dict['gen'] = beat_error.sum()



    return ret_dict,loss


from torch import Tensor
from torch.autograd import Variable



def calculate_gradient_penalty(real_data, fake_data, real_outputs, fake_outputs, k=2, p=6, device=torch.device("cuda")):
    """
    Calculates the gradient penalty loss for WGAN Div
    Adapted from: https://github.com/eriklindernoren/PyTorch-GAN
    @param real_data: the ground truth data
    @param fake_data: the generated data
    @param real_outputs: discriminator ground truth output
    @param fake_outputs: discriminator generated output
    @param k: k parameter of original implementation
    @param p: p parameter of original implementation
    @param device: cpu/gpu
    @return: gradient penalty
    """
    real_grad_outputs = torch.full((real_data.size(0),1), 1, dtype=torch.float32, requires_grad=False, device=device)
    fake_grad_outputs = torch.full((fake_data.size(0),1), 1, dtype=torch.float32, requires_grad=False, device=device)
    #real_data = real_data.reshape((real_data.shape[0], -1))
    #fake_data = fake_data.reshape((fake_data.shape[0], -1))

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


