import sys
import time

import wandb

from lookahead import Lookahead
from train_lightning import getWebDataSet, dt

[sys.path.append(i) for i in ['.', '..']]

from torch import optim
import torch.nn.functional as F
import matplotlib

from model.embedding_net import EmbeddingNet
from train_eval.train_joint_embed import eval_embed
from utils.average_meter import AverageMeter

matplotlib.use('Agg')  # we don't use interactive GUI

from config.parse_args import parse_args

from data_loader.lmdb_data_loader import *
import utils.train_utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def evaluate_testset(test_data_loader, generator):
    # to evaluation mode
    generator.train(False)

    losses = AverageMeter('loss')
    start = time.time()

    with torch.no_grad():
        for iter_idx, data in enumerate(test_data_loader, 0):
            time_seq, text_lengths, in_text_padded, _, target_vec, in_audio, audio_var_bottom, audio_var_top, in_spec, aux_info, annotation = data
            batch_size = target_vec.size(0)

            pre_poses = target_vec[:, :4, :].to(device)

            target = target_vec[:, 4:, :].to(device)
            audio_top = audio_var_top.to(device)
            audio_bottom = audio_var_bottom.to(device)

            in_audio = in_audio.to(device)
            in_text_padded = in_text_padded.to(device)

            # in_audio_top,in_audio_bottom,audio,in_text_padded, pre_poses, target_poses, net
            loss, _ = eval_embed(pre_poses, target, generator)
            losses.update(loss.item(), batch_size)

    # back to training mode
    generator.train(True)

    # print
    ret_dict = {'loss': losses.avg}
    elapsed_time = time.time() - start
    logging.info('[VAL] loss: {:.3f} / {:.1f}s'.format(losses.avg, elapsed_time))

    return ret_dict


def train_iter(args, epoch, target_data, audio_top, audio_bottom, audio, in_text_padded, net, optim):
    # zero gradients
    optim.zero_grad()

    variational_encoding = False  # AE or VAE

    pre_poses = target_data[:, :4, :].to(device)
    target = target_data[:, 4:, :].to(device)

    # reconstruction loss
    poses_feat, pose_mu, pose_logvar, recon_data = net(pre_poses, target, True)

    recon_loss = F.l1_loss(recon_data, target_data, reduction='none')
    recon_loss = torch.mean(recon_loss, dim=(1, 2))

    # if True:  # use pose diff
    #    target_diff = target[:, 1:] - target[:, :-1]
    #    recon_diff = recon_data[:, 1:] - recon_data[:, :-1]
    #    recon_loss += torch.mean(F.l1_loss(recon_diff, target_diff, reduction='none'), dim=(1, 2))

    recon_loss = torch.sum(recon_loss)

    # KLD
    if variational_encoding:
        KLD = -0.5 * torch.sum(1 + pose_logvar - pose_mu.pow(2) - pose_logvar.exp())

        if epoch < 10:
            KLD_weight = 0
        else:
            KLD_weight = min(1.0, (epoch - 10) * 0.05)
        recon_weight = 100
        loss = recon_weight * recon_loss + KLD_weight * KLD
    else:
        recon_weight = 1
        loss = recon_weight * recon_loss

    loss.backward()
    optim.step()

    ret_dict = {'loss': recon_weight * recon_loss.item()}
    if variational_encoding:
        ret_dict['KLD'] = KLD_weight * KLD.item()
    return ret_dict


def main(config):
    args = config['args']
    date_s = str(dt())

    wandb.login(key="")
    wandb.init(project='aqgt_feature')

    # random seed
    if args.random_seed >= 0:
        utils.train_utils.set_random_seed(args.random_seed)

    # set logger
    utils.train_utils.set_logger(args.model_save_path, os.path.basename(__file__).replace('.py', '.log'))

    # dataset
    mean_dir_vec = np.squeeze(np.array(args.mean_dir_vec))
    path = 'data/h36m/data_3d_h36m.npz'  # from https://github.com/facebookresearch/VideoPose3D/blob/master/DATASETS.md
    train_loader, val_loader, lang_model, speaker_model = getWebDataSet(args,
                                                                        ["dataset/AQGT/webdataset",
                                                                         "dataset/SaGA/webdataset"],
                                                                        "combined_vocab/")

    # train
    pose_dim = 159  # 53 x 3
    start = time.time()
    loss_meters = [AverageMeter('loss'), AverageMeter('var_loss')]
    best_val_loss = (1e+10, 0)  # value, epoch

    # interval params
    print_interval = 100
    save_sample_result_epoch_interval = 10
    save_model_epoch_interval = 20

    # init model and optimizer
    generator = EmbeddingNet(args, pose_dim, args.n_poses).to(device)
    gen_optimizer = optim.AdamW(generator.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    gen_optimizer = Lookahead(optimizer=gen_optimizer, k=6, alpha=0.5)
    gen_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(gen_optimizer, mode='min', factor=0.5, patience=8,
                                                           threshold=1e-10, min_lr=1e-5, verbose=False)
    # print("loading checkpoint", args.eval_net_path)
    # checkpoint = torch.load(args.eval_net_path, map_location="cpu")

    # print(generator.load_state_dict(checkpoint['gen_dict'], strict=True))

    # training
    global_iter = 0
    best_values = {}  # best values for all loss metrics
    for epoch in range(10000):
        wandb.log({"gen_lr": gen_optimizer.param_groups[0]['lr']})
        # evaluate the test set

        val_metrics = evaluate_testset(val_loader, generator)

        # best?
        val_loss = val_metrics['loss']
        wandb.log({"val_loss": val_metrics['loss']})
        gen_sched.step(val_metrics['loss'])
        is_best = val_loss < best_val_loss[0]
        if is_best:
            logging.info('  *** BEST VALIDATION LOSS: {:.3f}'.format(val_loss))
            best_val_loss = (val_loss, epoch)
        else:
            logging.info('  best validation loss so far: {:.3f} at EPOCH {}'.format(best_val_loss[0], best_val_loss[1]))

        # save model
        if is_best or (epoch % save_model_epoch_interval == 0 and epoch > 0):
            gen_state_dict = generator.state_dict()

            if is_best:
                os.makedirs(args.model_save_path + "_" + date_s, exist_ok=True)
                save_name = '{}/{}_checkpoint_best.bin'.format(args.model_save_path + "_" + date_s, args.name)
                utils.train_utils.save_checkpoint({
                    'args': args, 'epoch': epoch, 'pose_dim': pose_dim, 'gen_dict': gen_state_dict,
                }, save_name)
                if epoch > 0:
                    save_name = 'output/extractor_checkpoint_best.bin'.format(args.model_save_path, args.name)
                    utils.train_utils.save_checkpoint({
                        'args': args, 'epoch': epoch, 'pose_dim': pose_dim, 'gen_dict': gen_state_dict,
                    }, save_name)

        # save sample results
        # if args.save_result_video and epoch % save_sample_result_epoch_interval == 0:
        #    evaluate_sample_and_save_video(epoch, args.name, test_loader, generator, args=args)

        # train iter
        tbar = tqdm(enumerate(train_loader, 0), miniters=10)
        iter_start_time = time.time()
        for iter_idx, (
        time_seq, text_lengths, in_text_padded, _, target_vec, in_audio, audio_var_bottom, audio_var_top, in_spec,
        aux_info, annotation) in tbar:
            global_iter += 1
            batch_size = target_vec.size(0)

            audio_top = audio_var_top.to(device)
            audio_bottom = audio_var_bottom.to(device)
            in_audio = in_audio.to(device)
            in_text_padded = in_text_padded.to(device)

            loss = train_iter(args, epoch, target_vec.to(device), audio_top, audio_bottom, in_audio, in_text_padded,
                              generator, gen_optimizer)
            wandb.log(loss)
            # loss values
            for loss_meter in loss_meters:
                name = loss_meter.name
                if name in loss:
                    loss_meter.update(loss[name], batch_size)

            # print training status
            if (iter_idx + 1) % print_interval == 0:
                print_summary = 'EP {} ({:3d}) | {:>8s}, {:.0f} samples/s | '.format(
                    epoch, iter_idx + 1, utils.train_utils.time_since(start),
                           batch_size / (time.time() - iter_start_time))
                for loss_meter in loss_meters:
                    if loss_meter.count > 0:
                        print_summary += '{}: {:.3f}, '.format(loss_meter.name, loss_meter.avg)
                        loss_meter.reset()
                logging.info(print_summary)

            iter_start_time = time.time()

    # print best losses
    logging.info('--------- best loss values ---------')
    for key in best_values.keys():
        logging.info('{}: {:.3f} at EPOCH {}'.format(key, best_values[key][0], best_values[key][1]))


def evaluate_sample_and_save_video(epoch, prefix, test_data_loader, generator, args, n_save=None, save_path=None):
    generator.train(False)  # eval mode
    start = time.time()
    if not n_save:
        n_save = 1 if epoch <= 0 else 5

    with torch.no_grad():
        for iter_idx, data in enumerate(test_data_loader, 0):
            if iter_idx >= n_save:  # save N samples
                break

            time_seq, text_lengths, in_text_padded, _, target_vec, in_audio, audio_var_bottom, audio_var_top, in_spec, aux_info, entity_map, occurence_map, l_phase, r_phase, l_phrase, r_phrase = data

            # prepare
            select_index = 20
            target_dir_vec = target_dir_vec[select_index, :, :].unsqueeze(0).to(device)

            # generation
            _, _, _, _, _, _, out_dir_vec = generator(None, None, None, target_dir_vec, variational_encoding=False)

            # to video
            target_dir_vec = np.squeeze(target_dir_vec.cpu().numpy())
            out_dir_vec = np.squeeze(out_dir_vec.cpu().numpy())

            if save_path is None:
                save_path = args.model_save_path

            mean_data = np.array(args.mean_dir_vec).reshape(-1, 3)
            utils.train_utils.create_video_and_save(
                save_path, epoch, prefix, iter_idx,
                target_dir_vec, out_dir_vec, mean_data, '')

    generator.train(True)  # back to training mode
    logging.info('saved sample videos, took {:.1f}s'.format(time.time() - start))

    return True


if __name__ == '__main__':
    _args = parse_args()
    main({'args': _args})
