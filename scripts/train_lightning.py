import datetime
import itertools
import os.path
import pprint
import random
from collections import OrderedDict
from copy import copy
from random import shuffle

import cv2
import matplotlib
import torch.nn.functional as F
import torch.random
import wandb
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import TQDMProgressBar
from torch.utils.data import DataLoader
from torchelie.optim import RAdamW

from lookahead import Lookahead
from model.extra_losses import V7_loss
from model.AQGT_plus import AqgtPlusGenerator, MyLoopDiscriminator
from utils.SkeletonHelper.VisualizeHelper import VisualizeCvClass
from train_eval.train_AQGT_plus import train_dis, train_gen
from utils.data_utils import convert_dir_vec_to_pose
from utils.vocab_utils import build_vocab

# [sys.path.append(f) for f in ['.', '..']]

matplotlib.use('Agg')  # we don't use the interactive GUI

from config.parse_args import parse_args
from model.embedding_space_evaluator import EmbeddingSpaceEvaluator

from data_loader.lmdb_data_loader import *
import utils.train_utils

device = torch.device("cuda:0")


def filter_out_changed_layer(pretrained_dict, model_dict):
    input_size = len(list(pretrained_dict.keys()))
    p1 = {}
    for k, v in pretrained_dict.items():
        if k in model_dict:
            p1[k] = v
        else:
            print("removing layer:", k)
    pretrained_dict = p1
    if len(list(p1.keys())) != input_size:
        print("removed", input_size - len(list(p1.keys())), "non-existing layers.")

    input_size = len(list(pretrained_dict.keys()))
    p1 = OrderedDict()
    for k, v in pretrained_dict.items():
        if model_dict[k].shape == v.shape:
            p1[k] = v
        else:
            print("removed changed layer:", k)
    if len(list(p1.keys())) != input_size:
        print("removed", input_size - len(list(p1.keys())), "changed layers.")
    pretrained_dict = p1
    return pretrained_dict


class Lightning_Trainer(LightningModule):

    def __init__(self, args, train_dataloader, val_dataloader, speaker_model, lang_model):
        super().__init__()
        self.args = args
        self.learning_rate = args.learning_rate
        self.dis_opt_dict = None
        self.gen_opt_dict = None
        self.train_loader = train_dataloader
        self.val_loader = val_dataloader
        self.jerkLoss = V7_loss()
        self.speaker_model = speaker_model
        self.lang_model = lang_model
        self.hist_dict = {}
        self.epoch = 99
        os.environ["WDS_EPOCH"] = str(self.epoch)
        self.init_my_model(self.args, pose_dim=159, load=True)
        self.embed_space_evaluator = EmbeddingSpaceEvaluator(args, args.eval_net_path, lang_model, device)
        self.embed_space_evaluator_fast = EmbeddingSpaceEvaluator(args, args.eval_net_path, lang_model, device)
        self.mean_pose, self.mean_dir_vec = pickle.load(open("means.p", "rb"))
        self.automatic_optimization = True
        self.frechet = 10000
        self.save_date = dt()
        self.iteration_step = 0
        self.vis_done = 0

    def init_my_model(self, args, pose_dim, load=True):

        seq_pose_encoder = None

        if load and args.checkpoint_path != "None":
            print("loading checkpoint", args.checkpoint_path)
            checkpoint = torch.load(args.checkpoint_path, map_location="cpu")

            lang_m = checkpoint['lang_model'] if self.lang_model is None else self.lang_model
            speaker_m = checkpoint["speaker_model"] if self.speaker_model is None else self.speaker_model

            self.generator = AqgtPlusGenerator(args,
                                               n_words=lang_m.n_words,
                                               word_embed_size=args.wordembed_dim,
                                               word_embeddings=seq_pose_encoder,
                                               z_obj=speaker_m,
                                               pose_dim=pose_dim)
            self.discriminator = MyLoopDiscriminator(pose_dim, args, lang_m.n_words, args.wordembed_dim,
                                                     seq_pose_encoder)



            self.generator.load_state_dict(checkpoint['gen_dict'], strict=False)
            self.discriminator.load_state_dict(checkpoint['dis_dict'], strict=False)

            if "dis_opt_dict" in checkpoint:
                self.dis_opt_dict = checkpoint["dis_opt_dict"]
            if "gen_opt_dict" in checkpoint:
                self.gen_opt_dict = checkpoint["gen_opt_dict"]

        else:
            self.generator = AqgtPlusGenerator(args,
                                               n_words=self.lang_model.n_words,
                                               word_embed_size=args.wordembed_dim,
                                               word_embeddings=seq_pose_encoder,
                                               z_obj=self.speaker_model,
                                               pose_dim=pose_dim)
            self.discriminator = MyLoopDiscriminator(pose_dim, args, self.lang_model.n_words, args.wordembed_dim,
                                                     seq_pose_encoder)

        return self.generator, self.discriminator


    def configure_optimizers(self):
        print(self.learning_rate)
        self.beat_gen_optimizer = RAdamW(self.generator.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        self.beat_gen_optimizer = Lookahead(optimizer=self.beat_gen_optimizer, k=6, alpha=0.5)

        self.beat_dis_optimizer = RAdamW(self.discriminator.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        self.beat_dis_optimizer = Lookahead(optimizer=self.beat_dis_optimizer, k=6, alpha=0.5)

        if self.gen_opt_dict is not None:
            print("loading optimizer")
            self.beat_gen_optimizer.load_state_dict(self.gen_opt_dict)
            self.beat_dis_optimizer.load_state_dict(self.dis_opt_dict)

        return (
            {'optimizer': self.beat_gen_optimizer, 'frequency': 1},
            {'optimizer': self.beat_dis_optimizer, 'frequency': 1},
        )

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def training_step(self, batch, batch_idx, optimizer_idx):

        time_seq, text_lengths, in_text_padded, _, target_vec, in_audio, audio_var_bottom, audio_var_top, in_spec, aux_info, annotation = batch

        ret_dict = {}

        vids = aux_info
        vid_indices = [self.speaker_model.word2index[vid] if vid in self.speaker_model.word2index.keys() else self.speaker_model.getAddIndex(vid) for vid in vids]
        vid_indices = torch.LongTensor(vid_indices).to(device)
        if optimizer_idx == 0:
            self.iteration_step += 1
            ret_dict, loss = train_gen(self.generator, target_vec, self.args, time_seq, in_text_padded, in_audio,
                                       vid_indices, self.discriminator, self.jerkLoss, ret_dict, self.epoch,
                                       audio_var_bottom, audio_var_top, self.args.loss_warmup, annotation,
                                       train_gan=self.iteration_step % 3 == 0 and self.epoch > self.args.loss_warmup)
        else:
            ret_dict, loss = train_dis(self.generator, time_seq, in_text_padded, in_audio, vid_indices,
                                       self.discriminator, target_vec, ret_dict, self.args, audio_var_bottom,
                                       audio_var_top, annotation)

        loss_out = self.createWandbLog(ret_dict)
        if self.args.wandb_key != "":
            wandb.log(loss_out)

        return loss

    def on_validation_end(self) -> None:
        save_folder = "output/" + str(self.save_date) + "/"
        os.makedirs(save_folder, exist_ok=True)
        save_name = save_folder + '{:.8f}_{}_checkpoint.bin'.format(self.frechet, self.epoch)
        if self.frechet < self.frechet_fast:
            print("using slow weights")
            self.beat_gen_optimizer._backup_and_load_cache()
        utils.train_utils.save_checkpoint({
            'args': self.args, 'epoch': self.epoch, 'lang_model': self.lang_model, 'speaker_model': self.speaker_model,
            'pose_dim': 159, 'gen_dict': self.generator.state_dict(),
            'dis_dict': self.discriminator.state_dict(), "dis_opt_dict": self.beat_dis_optimizer.state_dict(),
            'gen_opt_dict': self.beat_gen_optimizer.state_dict(),
            "dis_opt2_dict": self.beat_dis_optimizer.optimizer.state_dict(),
            'gen_opt2_dict': self.beat_gen_optimizer.optimizer.state_dict()
        }, save_name)
        if self.frechet < self.frechet_fast:
            self.beat_gen_optimizer._clear_and_load_backup()

    def do_validation(self, batch, embed_space):
        time_seq, text_lengths, in_text_padded, _, target_vec, in_audio, audio_var_bottom, audio_var_top, in_spec, aux_info, annotation = batch

        pre_seq = target_vec.new_zeros((target_vec.shape[0], target_vec.shape[1], target_vec.shape[2] + 1)).to(device)
        pre_seq[:, 0:self.args.n_pre_poses, :-1] = target_vec[:, 0:self.args.n_pre_poses]
        pre_seq[:, 0:self.args.n_pre_poses, -1] = 1

        vid_indices = [random.choice(list(self.speaker_model.word2index.values())) for _ in range(target_vec.shape[0])]
        vid_indices = torch.LongTensor(vid_indices).to(device)

        out_dir_vec, *_ = self.generator(pre_seq, time_seq, in_text_padded, in_audio, audio_var_bottom, audio_var_top,
                                         vid_indices, annotation)
        loss = F.l1_loss(out_dir_vec, target_vec)
        embed_space.push_samples(out_dir_vec, target_vec)

        j1, j2, j3, j4, j5, j6, j7 = self.jerkLoss(out_dir_vec, target_vec)

        # calculate MAE of joint coordinates
        out_dir_vec = out_dir_vec.cpu().numpy()

        mean_vec = self.mean_dir_vec.flatten()
        out_dir_vec += mean_vec
        out_joint_poses = convert_dir_vec_to_pose(out_dir_vec)
        target_vec = target_vec.cpu().numpy()
        target_vec += mean_vec
        target_poses = convert_dir_vec_to_pose(target_vec)

        if out_joint_poses.shape[1] == self.args.n_poses:
            diff = out_joint_poses[:, self.args.n_pre_poses:] - target_poses[:, self.args.n_pre_poses:]
        else:
            diff = out_joint_poses - target_poses[:, self.args.n_pre_poses:]
        mae_val = np.mean(np.absolute(diff))

        ret_dict = {'loss': loss, 'joint_mae': mae_val, 'velocity_loss': j1, "acceleration_loss": j2, "jerk_loss": j3}
        ret_dict['jounce_loss'] = j4
        ret_dict['crackle_loss'] = j5
        ret_dict['pop_loss'] = j6
        ret_dict['lock_loss'] = j7

        if self.vis_done < 16:
            self.vis_done += 1
            if self.args.wandb_key != "":
                wandb.log({"val_video_" + str(self.vis_done): self.vis_difference(out_joint_poses[0], target_poses[0])})

        return ret_dict

    def vis_difference(self, output_poses, target_poses):
        vis = VisualizeCvClass()
        comb_imgs = []
        for i in tqdm(range(output_poses.shape[0])):
            f1 = vis.VisUpperBody(output_poses[i].reshape((53, 3)), np.ones(53, ))
            f2 = vis.VisUpperBody(target_poses[i].reshape((53, 3)), np.ones(53, ))
            cv2.putText(f1, 'Generated:', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, 2)
            cv2.putText(f2, 'Human:', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, 2)
            img = np.concatenate((f2, f1), axis=1)
            comb_imgs.append(np.moveaxis(img, 2, 0))
        comb_imgs = np.asarray(comb_imgs)
        del vis
        return wandb.Video(comb_imgs, fps=15, format="mp4")

    def validation_step(self, batch, batch_idx):

        self.do_validation(batch, self.embed_space_evaluator_fast)
        self.beat_gen_optimizer._backup_and_load_cache()
        ret2 = self.do_validation(batch, self.embed_space_evaluator)
        self.beat_gen_optimizer._clear_and_load_backup()
        return ret2

    def validation_epoch_end(self, outputs):
        d_report = {}
        keys = outputs[0].keys()
        for k in keys:
            avg = torch.stack([torch.tensor(x[k]) for x in outputs]).mean()
            d_report["val_" + k] = avg

        frechet_dist, feat_dist = self.embed_space_evaluator.get_scores()
        frechet_dist_fast, feat_dist_fast = self.embed_space_evaluator_fast.get_scores()
        self.embed_space_evaluator.reset()
        self.embed_space_evaluator_fast.reset()

        avg_loss = frechet_dist
        self.frechet = frechet_dist
        self.frechet_fast = frechet_dist_fast
        d_report["epoch"] = self.epoch
        d_report['frechet'] = frechet_dist
        d_report['frechet_fast'] = frechet_dist_fast
        d_report['feat_dist'] = feat_dist
        for param_group in self.beat_gen_optimizer.optimizer.param_groups:
            d_report['gen_lr'] = param_group['lr']
        for param_group in self.beat_dis_optimizer.optimizer.param_groups:
            d_report['dis_lr'] = param_group['lr']
        if self.args.wandb_key != "":
            wandb.log(d_report)
        self.epoch += 1

        os.environ["WDS_EPOCH"] = str(self.epoch)
        self.log("val_loss", avg_loss, sync_dist=True, prog_bar=True)
        self.vis_done = 0

        return {"val_loss": avg_loss}

    def createWandbLog(self, loss_out):
        out = {}
        for k in loss_out.keys():
            hist = []
            if k in self.hist_dict:
                hist = self.hist_dict[k]
            hist.append(loss_out[k].cpu().detach().numpy())
            hist = hist[-500:]
            out[k] = np.median(np.asarray(hist))
            self.hist_dict[k] = hist
        return out


def makeWebDataSet(args, beat_path, mean_dir_vec, mean_pose, collate_fn):
    assert len(args.train_data_path) == len(args.val_data_path) == len(args.test_data_path) == len(args.web_data_path)

    for (train_path,val_path,test_path,webdataset_path) in zip(args.train_data_path,args.val_data_path,args.test_data_path,args.web_data_path):
        if not os.path.exists(webdataset_path) or len(os.listdir()) < 1:
            train_dataset = SpeechMotionDataset(train_path, beat_path=beat_path,
                                                n_poses=args.n_poses,
                                                subdivision_stride=args.subdivision_stride,
                                                pose_resampling_fps=args.motion_resampling_framerate,
                                                mean_dir_vec=mean_dir_vec,
                                                mean_pose=mean_pose,
                                                remove_word_timing=(args.input_context == 'text'),
                                                beat_generating=True)
            val_dataset = SpeechMotionDataset(val_path, beat_path=beat_path,
                                              n_poses=args.n_poses,
                                              subdivision_stride=args.subdivision_stride,
                                              pose_resampling_fps=args.motion_resampling_framerate,
                                              speaker_model=train_dataset.speaker_model,
                                              mean_dir_vec=mean_dir_vec,
                                              mean_pose=mean_pose,
                                              remove_word_timing=(args.input_context == 'text'),
                                              beat_generating=True
                                              )
            test_dataset = SpeechMotionDataset(test_path, beat_path=beat_path,
                                               n_poses=args.n_poses,
                                               subdivision_stride=args.subdivision_stride,
                                               pose_resampling_fps=args.motion_resampling_framerate,
                                               speaker_model=train_dataset.speaker_model,
                                               mean_dir_vec=mean_dir_vec,
                                               mean_pose=mean_pose,
                                               beat_generating=True)
            vocab_cache_path = os.path.join("combined_vocab", 'vocab_cache.pkl')
            lang_model = build_vocab('words', [train_dataset, val_dataset, test_dataset], vocab_cache_path,
                                     args.wordembed_path,
                                     args.wordembed_dim)
            train_dataset.set_lang_model(lang_model)
            val_dataset.set_lang_model(lang_model)
            test_dataset.set_lang_model(lang_model)

            train_loader = DataLoader(dataset=train_dataset, batch_size=1,
                                      shuffle=False, drop_last=True, num_workers=4,
                                      collate_fn=collate_fn
                                      )

            val_loader = DataLoader(dataset=val_dataset, batch_size=1,
                                    shuffle=False, drop_last=True, num_workers=4,
                                    collate_fn=collate_fn
                                    )

            test_loader = DataLoader(dataset=test_dataset, batch_size=1,
                                    shuffle=False, drop_last=True, num_workers=4,
                                    collate_fn=collate_fn
                                    )
            os.makedirs(webdataset_path + "/train", exist_ok=False)
            os.makedirs(webdataset_path + "/val", exist_ok=False)
            os.makedirs(webdataset_path + "/test", exist_ok=False)
            import webdataset as wds
            sink = wds.ShardWriter(webdataset_path + "/train/data_%06d.tar", maxcount=128 * 20, compress=False)
            for iter_idx, data in tqdm(enumerate(train_loader), total=train_loader.__len__()):
                beats, word_seq, words_lengths, text_padded, poses_seq, vec_seq, audio, audio_var_bottom, audio_var_top, spectrogram, aux_info, annotations = data
                sink.write({
                    "__key__": "sample%06d" % iter_idx,
                    "beat.pth": beats[0].cpu().detach(),
                    "word_seq.pth": word_seq[0].cpu().detach(),
                    "words_lengths.pth": words_lengths.cpu().detach(),
                    "text_padded.pth": text_padded[0].cpu().detach(),
                    "poses_seq.pth": poses_seq[0].cpu().detach(),
                    "vec_seq.pth": vec_seq[0].cpu().detach(),
                    "audio.pth": audio[0].cpu().detach(),
                    "audio_var_bottom.pth": audio_var_bottom[0].cpu().detach(),
                    "audio_var_top.pth": audio_var_top[0].cpu().detach(),
                    "spectrogram.pth": spectrogram[0].cpu().detach(),
                    "aux_info.pth": aux_info,
                    "annotations.pth": annotations[0].cpu().detach() if annotations is not None else None,
                })

            sink.close()
            sink = wds.ShardWriter(webdataset_path + "/val/data_%06d.tar", maxcount=128 * 20, compress=False)
            for iter_idx, data in tqdm(enumerate(val_loader), total=val_loader.__len__()):
                beats, word_seq, words_lengths, text_padded, poses_seq, vec_seq, audio, audio_var_bottom, audio_var_top, spectrogram, aux_info, annotations = data
                sink.write({
                    "__key__": "sample%06d" % iter_idx,
                    "beat.pth": beats[0].cpu().detach(),
                    "word_seq.pth": word_seq[0].cpu().detach(),
                    "words_lengths.pth": words_lengths.cpu().detach(),
                    "text_padded.pth": text_padded[0].cpu().detach(),
                    "poses_seq.pth": poses_seq[0].cpu().detach(),
                    "vec_seq.pth": vec_seq[0].cpu().detach(),
                    "audio.pth": audio[0].cpu().detach(),
                    "audio_var_bottom.pth": audio_var_bottom[0].cpu().detach(),
                    "audio_var_top.pth": audio_var_top[0].cpu().detach(),
                    "spectrogram.pth": spectrogram[0].cpu().detach(),
                    "aux_info.pth": aux_info,
                    "annotations.pth": annotations[0].cpu().detach() if annotations is not None else None,
                })
            sink.close()

            sink = wds.ShardWriter(webdataset_path + "/test/data_%06d.tar", maxcount=128 * 20, compress=False)
            for iter_idx, data in tqdm(enumerate(test_loader), total=test_loader.__len__()):
                beats, word_seq, words_lengths, text_padded, poses_seq, vec_seq, audio, audio_var_bottom, audio_var_top, spectrogram, aux_info, annotations = data
                sink.write({
                    "__key__": "sample%06d" % iter_idx,
                    "beat.pth": beats[0].cpu().detach(),
                    "word_seq.pth": word_seq[0].cpu().detach(),
                    "words_lengths.pth": words_lengths.cpu().detach(),
                    "text_padded.pth": text_padded[0].cpu().detach(),
                    "poses_seq.pth": poses_seq[0].cpu().detach(),
                    "vec_seq.pth": vec_seq[0].cpu().detach(),
                    "audio.pth": audio[0].cpu().detach(),
                    "audio_var_bottom.pth": audio_var_bottom[0].cpu().detach(),
                    "audio_var_top.pth": audio_var_top[0].cpu().detach(),
                    "spectrogram.pth": spectrogram[0].cpu().detach(),
                    "aux_info.pth": aux_info,
                    "annotations.pth": annotations[0].cpu().detach() if annotations is not None else None,
                })
            sink.close()

            del train_loader
            del val_loader
            del test_loader


func_lang_model = None


def getWebDataSet(args, web_paths, vocab_cache_path):
    global func_lang_model
    import webdataset as wds
    from webdataset import WebLoader

    os.environ["WDS_EPOCH"] = str(0)

    # vocab_cache_path = os.path.join(vocab_cache_path, 'vocab_cache.pkl')

    with open(vocab_cache_path + "/" + 'dataset_train_speaker_model.pkl', 'rb') as f:
        speaker_model = pickle.load(f)

    lang_model = build_vocab('words', None, os.path.join(vocab_cache_path, 'vocab_cache.pkl'),
                             args.wordembed_path,
                             args.wordembed_dim)
    from torch import nn
    embedding = nn.Embedding.from_pretrained(torch.FloatTensor(lang_model.word_embedding_weights), freeze=True)
    func_lang_model = embedding
    trainlists = []
    vallists = []

    for webdataset_path in web_paths:
        traintemp = []
        for root, dirs, files in os.walk(webdataset_path + "/train/"):
            shuffle(files)
            for file in tqdm(files):
                filename = os.path.join(root, file)
                if filename.endswith(".tar"):
                    traintemp.append(filename)
        valtemp = []
        for root, dirs, files in os.walk(webdataset_path + "/val/"):
            shuffle(files)
            for file in tqdm(files):
                filename = os.path.join(root, file)
                if filename.endswith(".tar"):
                    valtemp.append(filename)
        trainlists.append(traintemp)
        vallists.append(valtemp)

    max_train = max([len(f) for f in trainlists])
    max_val = max([len(f) for f in vallists])
    print("max train list length", max_train)
    print("max val   list length", max_val)

    def fill_list(trainlist, max_v):
        verbose = False
        templist = copy(trainlist)
        if len(trainlist) != max_v:
            print("list length is at", len(trainlist), "of", max_v)
            print("filling...")
            verbose = True
        while len(trainlist) < (max_v if max_v <= 10 else max_v // 3):
            trainlist.append(random.choice(templist))
        if verbose:
            print("list length is at", len(trainlist), "of", max_v)
        return trainlist

    for i in range(len(trainlists)):
        trainlists[i] = fill_list(trainlists[i], max_train)
    # for i in range(len(vallists)):
    #    vallists[i] = fill_list(vallists[i],max_val)

    trainlist = list(itertools.chain.from_iterable(trainlists))
    vallist = list(itertools.chain.from_iterable(sorted(vallists)))

    # trainlist = trainlist[:1]
    # vallist = vallist[:1]

    shardlist_train = wds.PytorchShardList(trainlist, epoch_shuffle=True)
    train_dataset1 = wds.WebDataset(shardlist_train, shardshuffle=True).decode() \
        .to_tuple("beat.pth", "words_lengths.pth", "word_seq.pth",
                  "poses_seq.pth", "vec_seq.pth", "audio.pth", "audio_var_bottom.pth", "audio_var_top.pth",
                  "spectrogram.pth",
                  "aux_info.pth", "annotations.pth") \
        .map_tuple(identity, identity, identity, identity, identity, identity, identity, identity, identity, tolist,
                   remove_corrupt)

    train_loader = WebLoader(train_dataset1, num_workers=args.loader_workers, batch_size=args.batch_size,
                             pin_memory=False, persistent_workers=False).unbatched().shuffle(
        args.batch_size * 30).batched(args.batch_size)

    shardlist_val = wds.PytorchShardList(vallist, epoch_shuffle=False)
    val_dataset = wds.WebDataset(shardlist_val, shardshuffle=False).decode() \
        .to_tuple("beat.pth", "words_lengths.pth", "word_seq.pth",
                  "poses_seq.pth", "vec_seq.pth", "audio.pth", "audio_var_bottom.pth", "audio_var_top.pth",
                  "spectrogram.pth",
                  "aux_info.pth", "annotations.pth") \
        .map_tuple(identity, identity, identity, identity, identity, identity, identity, identity, identity, tolist,
                   remove_corrupt)

    val_loader = WebLoader(val_dataset, num_workers=args.loader_workers, batch_size=args.batch_size, pin_memory=False,
                           persistent_workers=False)

    return train_loader, val_loader, lang_model, speaker_model
    # build vocab


def identity(x):
    return x


def remove_corrupt(x):
    if torch.is_tensor(x) and len(x.shape) == 2 and x.shape[0] == 17 and x.shape[1] == 34:
        return x
    else:
        return torch.zeros((17, 34)) - 1


def tolist(x):
    return x['vid'][0]


def totext(x):
    return x['word_list'][0]


def dt():
    return datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")

def change_weights(state_dict):
    out = OrderedDict()
    change_k = [["gru.","transformer_1."],
                ["gru2.","transformer_2."],
                ["gru3.", "gru_2."],
                ]
    for k,v in state_dict.items():
        k1 = k
        #print(k)
        for b1,b2 in change_k:
            if k.startswith(b1):
                k = k.replace(b1,b2)
        if k != k1:
            print("changed:",k)
        out[k] = v
    return out

def removeAnno(state_dict):
    out = OrderedDict()
    for k, v in state_dict.items():
        if not "anno" in k:
            out[k] = v
        else:
            print(k)
    return out

def load_lightning_model(args, path):
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")

    lang_model = checkpoint["lang_model"]
    speaker_model = checkpoint["speaker_model"]
    out_dim = checkpoint["pose_dim"]

    model = Lightning_Trainer(args=args, train_dataloader=None, val_dataloader=None, speaker_model=speaker_model,
                              lang_model=lang_model)


    args = model.args
    generator = model.generator
    loss_fn = None

    return args, generator, loss_fn, lang_model, speaker_model, out_dim


def main(config):
    args = config['args']
    args.model_save_path = args.model_save_path + '_' + dt()

    if args.wandb_key != "":
        wandb.login(key=args.wandb_key)
        wandb.init(project='AQGT')


    # fixed seed?
    if args.random_seed >= 0:
        utils.train_utils.set_random_seed(args.random_seed)
        torch.random.seed(args.random_seed)

    # set logger
    utils.train_utils.set_logger(args.model_save_path, os.path.basename(__file__).replace('.py', '.log'))

    logging.info("PyTorch version: {}".format(torch.__version__))
    logging.info("CUDA version: {}".format(torch.version.cuda))
    logging.info("{} GPUs, default {}".format(torch.cuda.device_count(), device))
    logging.info(pprint.pformat(vars(args)))

    collate_fn = word_seq_add_beat_collate_fn

    # dataset
    mean_pose, mean_dir_vec = pickle.load(open("means.p", "rb"))
    beat_path = args.beat_data_path[0] + "/"

    #combine_speech_vocab_model(args, ["dataset/AQGT/dataset_train_speaker_model.pkl",
    #                                  "dataset/SaGA/dataset_train_speaker_model.pkl"],
    #                           "combined_vocab/dataset_train_speaker_model.pkl", fill_word_vectors=True)
    #combine_speech_vocab_model(args, ["dataset/AQGT/vocab_cache.pkl", "dataset/SaGA/vocab_cache.pkl"],
    #                           "combined_vocab/vocab_cache.pkl", fill_word_vectors=True)

    makeWebDataSet(args, beat_path, mean_dir_vec, mean_pose, collate_fn)

    train_loader, val_loader, lang_model, speaker_model = getWebDataSet(args, args.web_data_path,"combined_vocab/")
    model = Lightning_Trainer(args, train_loader, val_loader, speaker_model, lang_model)
    trainer = Trainer(
        accelerator="auto",
        devices=1,  # limiting got iPython runs
        max_epochs=args.epochs,
        amp_backend="apex",
        amp_level='O2',
        gradient_clip_val=0.5, #required for the wgan discriminator. Otherwise the training can produce nan values.
        callbacks=[TQDMProgressBar(refresh_rate=1)],  # val_check_interval=1000,
        benchmark=True, num_sanity_val_steps=2, detect_anomaly=False
    )
    trainer.fit(model)


if __name__ == '__main__':
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.multiprocessing.set_start_method('spawn')
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    _args = parse_args()
    main({'args': _args})
