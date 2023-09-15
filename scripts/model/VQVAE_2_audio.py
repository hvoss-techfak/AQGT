

import os
import random
import traceback
from random import shuffle

# os.environ["SLURM_JOB_NAME"] = "bash"
import numpy as np
import torch
import wandb
import webdataset as wds
# transforms = A.Compose(
#         [
#             A.CoarseDropout(max_holes=8,max_width=0.1,max_height=0.1, fill_value=0, p=1),
#             #A.OneOf([
#             #    A.OpticalDistortion(p=0.3),
#             #    A.GridDistortion(p=.1),
#             #    A.IAAPiecewiseAffine(p=0.3),
#             #], p=0.2),
#             #A.OneOf([
#             #    A.IAASharpen(),
#             #    A.IAAEmboss(),
#             #    A.RandomBrightnessContrast(),
#             #], p=0.3),
#             A.OneOf([
#                 A.IAAAdditiveGaussianNoise(),
#                 A.GaussNoise(),
#             ], p=0.5),
#             #A.OneOf([
#             #    A.MotionBlur(p=.2),
#             #    A.Blur(blur_limit=3, p=0.1),
#             #], p=0.2),
#         ])
from scripts.model.vq.VQVAE import VQVAE
from moviepy.video.io.VideoFileClip import VideoFileClip
from pedalboard_native import Chorus, Reverb, Bitcrush, Compressor, Delay, Distortion, \
    HighpassFilter, LowpassFilter, MP3Compressor, PeakFilter, Phaser, PitchShift
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core import LightningModule
from scipy.io import wavfile
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from webdataset import WebLoader

effects = [Chorus(rate_hz=1),Chorus(rate_hz=3),Chorus(rate_hz=6),Chorus(rate_hz=10),
               Reverb(room_size=0.25),Reverb(room_size=0.5),Reverb(room_size=1),
               Bitcrush(bit_depth=4),Bitcrush(bit_depth=5),Bitcrush(bit_depth=6),Bitcrush(bit_depth=7),Bitcrush(bit_depth=8),
               Compressor(),
               Delay(delay_seconds=0.1),
               Distortion(drive_db=5),Distortion(drive_db=10),Distortion(drive_db=15),Distortion(drive_db=20),
               HighpassFilter(50),HighpassFilter(200),HighpassFilter(500),HighpassFilter(1000),HighpassFilter(1500),
               LowpassFilter(50),LowpassFilter(200),LowpassFilter(500),LowpassFilter(1000),LowpassFilter(1500),
               MP3Compressor(0.2),MP3Compressor(0.5),MP3Compressor(1.0),MP3Compressor(2.0),
               PeakFilter(cutoff_frequency_hz=200,gain_db=-20),PeakFilter(cutoff_frequency_hz=500,gain_db=-20),
               PeakFilter(cutoff_frequency_hz=1200,gain_db=-20),PeakFilter(cutoff_frequency_hz=2400,gain_db=-20),
               PeakFilter(cutoff_frequency_hz=4800,gain_db=-20),PeakFilter(cutoff_frequency_hz=9600,gain_db=-20),
               Phaser(rate_hz=0.5),Phaser(rate_hz=2),Phaser(rate_hz=5),Phaser(rate_hz=10),Phaser(rate_hz=20),
               PitchShift(-12),PitchShift(-6),PitchShift(-2),PitchShift(2),PitchShift(6),PitchShift(12)]

class VQ_VAE_2_audio(LightningModule):

    def __init__(self,
                 config,epochs=100,epochs_done=0,learning_rate=0.01, **kwargs):
        os.environ["WDS_EPOCH"] = str(0)
        wandb.init(project='audio_training_vq_vae')
        wandb.config.update(config)
        super().__init__()
        self.save_hyperparameters()
        self.shardsize = 1557
        self.data_len = (4000 * self.shardsize) + (501 * self.shardsize)
        self.latent_dim = 128
        self.lr = learning_rate
        self.config = config
        self.b1 = 0.5
        self.b2 = 0.999
        self.batch_size = 128
        self.epoch_num = 0
        self.epochs = epochs
        self.init = False
        self.epochs_done = epochs_done
        os.environ["WDS_EPOCH"] = str(self.epoch_num)
        # networks
        mnist_shape = (1, 64, 64)
        self.hist_dict = {}
        self.generator = VQVAE(in_channel=1,channel=256,embed_dim=1,n_res_channel=256)
        self.latent_loss_weight = 0.25
        #self.discriminator = Discriminator(img_shape=mnist_shape)


    def forward(self, *args, **kwargs):
        pass



    def training_step(self, batch, batch_idx):

        quant_t_a, quant_b_a, diff_a, _, _ = self.generator.encode(batch[0])
        dec = self.generator.decode(quant_t_a, quant_b_a)
        recon_loss = F.smooth_l1_loss(dec, batch[1])
        latent_loss = diff_a.mean()

        loss = recon_loss + self.latent_loss_weight * latent_loss

        loss_out = {}
        loss_out["vqvae_loss"] = recon_loss + self.latent_loss_weight * latent_loss
        loss_out["vqvae_recon_loss"] = recon_loss
        loss_out["vqvae_latent_loss"] = latent_loss

        loss_out = self.createWandbLog(loss_out)

        wandb.log(loss_out)

        return loss


    def createWandbLog(self,loss_out):
        out = {}
        for k in loss_out.keys():
            hist = []
            if k in self.hist_dict:
                hist = self.hist_dict[k]
            hist.append(loss_out[k].cpu().detach().numpy())
            hist = hist[-3000:]
            out[k] = np.median(np.asarray(hist))
            self.hist_dict[k] = hist
        return out


    def validation_step(self, batch, batch_idx):
        quant_t_a, quant_b_a, diff_a, _, _ = self.generator.encode(batch[0])
        dec = self.generator.decode(quant_t_a, quant_b_a)
        recon_loss = F.smooth_l1_loss(dec, batch[1])
        latent_loss = diff_a.mean()

        loss_out = {}
        loss_out["vqvae_loss"] = recon_loss + self.latent_loss_weight * latent_loss
        loss_out["vqvae_recon_loss"] = recon_loss
        loss_out["vqvae_latent_loss"] = latent_loss

        return loss_out

    def validation_epoch_end(self, outputs):
        d_report = {}
        keys = outputs[0].keys()
        for k in keys:
            avg = torch.stack([torch.tensor(x[k]) for x in outputs]).mean()
            d_report["val_" + k] = avg
        avg_loss = torch.stack([x['vqvae_loss'] for x in outputs]).mean()
        d_report["epoch"] = self.epoch_num
        wandb.log(d_report)
        self.epoch_num += 1
        os.environ["WDS_EPOCH"] = str(self.epoch_num)
        self.inferenceAudio()
        self.log("val_loss",avg_loss,sync_dist=True,prog_bar=True)
        self.scheduler1.step(avg_loss)
        try:
            pass
            #self.inferenceAudio("SaGA", "out.mp4")
            #self.inferenceAudio("turtle", "out2.mp4")
            #self.inferenceAudio("shelter", "out3.mp4")
        except Exception as e:
            traceback.print_exc()
        return {"val_loss": avg_loss}

    def inferenceAudio(self, name, file):
        with torch.no_grad():
            video = VideoFileClip(file, audio_fps=16000)
            audio = video.audio
            arr = audio.to_soundarray()
            if len(arr.shape) > 1:
                arr = arr.mean(axis=1)
            prunelength = (arr.shape[0] // 4000) * 4000

            arr = arr[:prunelength]
            arr -= arr.min()
            arr /= arr.max() / 2
            arr -= 1
            arr = arr.reshape((-1, 4000))
            out_arr = np.zeros((arr.shape[0], 4096))
            out_arr[:, :4000] = arr
            out_arr = out_arr.reshape((-1, 1, 64, 64))
            print(out_arr.shape)
            for i in tqdm(range(out_arr.shape[0])):
                quant_t, quant_b, diff_a, _, _ = self.generator.encode(torch.from_numpy(out_arr[i, :, :, :]).unsqueeze(0).float().cuda())
                dec = self.generator.decode(quant_t, quant_b)
                out_arr[i, :, :, :] = dec.cpu().detach().numpy()
            out_arr = out_arr.reshape((-1, 4096))[:, :4000]
            out_arr = out_arr.flatten()
            audio_out = wandb.Audio(out_arr, sample_rate=16000)
            wavfile.write("audio.wav",16000,out_arr)
            wandb.log({name: audio_out})
            audio.close()
            video.close()
            del arr
            del out_arr

    def configure_optimizers(self):

        b1 = self.b1
        b2 = self.b2

        opt_g = torch.optim.AdamW(list(self.LogVAE_top.parameters()) + list(self.LogVAE_bottom.parameters()) + list(self.generator.parameters()), lr=1.3345453405760623e-05, betas=(b1, b2))

        scheduler1 = ReduceLROnPlateau(opt_g, mode='min', factor=0.5, patience=5, threshold=0.00000001)

        self.scheduler1 = scheduler1

        self.optimizer1 = opt_g
        return (
            {'optimizer': opt_g, 'frequency': 1,'scheduler':scheduler1},
        )

    def train_dataloader(self):

        folder = "/run/user/1000/gvfs/smb-share:server=moleman,share=hendric/Trimodal_audio/train"
        return self.getLoaderfromFolde(folder, noaugment=False)

    def val_dataloader(self):
        folder = "/run/user/1000/gvfs/smb-share:server=moleman,share=hendric/Trimodal_audio/val"
        return self.getLoaderfromFolde(folder, noaugment=True)

    def getLoaderfromFolde(self, folder, noaugment=False):
        filelist = []
        for root, dirs, files in os.walk(folder):
            shuffle(files)
            for file in tqdm(files):
                filename = os.path.join(root, file)
                if filename.endswith(".tar"):
                    filelist.append(filename)
        print(len(filelist))
        shuffle(filelist)
        shardlist = wds.PytorchShardList(filelist, epoch_shuffle=True)
        mapfunction = changeAudio_noaugment
        train_dataset = wds.WebDataset(shardlist, shardshuffle=True).decode().to_tuple("audio_out.npy","audio_out.npy").map_tuple(changeAudio_augment,changeAudio_noaugment)
        loader = WebLoader(train_dataset, num_workers=8, batch_size=self.batch_size).unbatched().shuffle(self.batch_size*10).batched(self.batch_size)
        return loader


def identity(x):
    return x

def changeAudio_noaugment(x):
    if x.flatten().shape[0] != 4096:
        out = np.zeros((4096,), dtype=np.float32)
        out[0:x.shape[0]] = x
    else:
        out = x
    out = out.reshape((64, 64, 1))
    out = torch.from_numpy(out).float()
    out = torch.moveaxis(out, 2, 0)

    return out

def changeAudio_augment(x):

    for i in range(1):
        x = random.choice(effects)(x, 16000)

    if x.flatten().shape[0] != 4096:
        out = np.zeros((4096,), dtype=np.float32)
        out[0:x.shape[0]] = x
    else:
        out = x
    out = out.reshape((64, 64, 1))
    out = torch.from_numpy(out).float()
    out = torch.moveaxis(out, 2, 0)
    return out

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def train(config, num_epochs=10, num_gpus=0):
    if config is None:
            config = {
              "lr": 0.0004,
              "pkt": 0.27,
              "div1": 30,
              "div2": 528971237.5043111,
              "noise_r": 119.25521201678656,
              "reg_weight": 0.002,
              "augment_percent_one":5.1307517955899575,
              "augment_percent_two":58.52219585159578,
              "use_l2_mse":True
            }

    checkpoints = ModelCheckpoint(
            dirpath="checkpoint_audio/",
            filename="{val_loss:10f}-{epoch}",
            monitor="val_loss",
            save_top_k=-1
        )
    model = VQ_VAE_2_audio(config, num_epochs)
    trainer = Trainer(
        # logger=logger,
        gpus=1,
        #num_nodes=1,
        #accelerator="ddp",
        #callbacks=[TuneReportCallback(metrics, on="validation_end")],
        callbacks=[checkpoints,],
        default_root_dir="./models/",
        max_epochs=num_epochs,
        amp_backend="native",
        gradient_clip_val=0.5,
        stochastic_weight_avg=True,
        benchmark=True,

    )
    trainer.fit(model)


if __name__ == "__main__":
    model = VQ_VAE_2_audio.load_from_checkpoint("pretrained/vqvae_audio/vqvae_audio.ckpt",strict=False).cuda()
    model.inferenceAudio("out2","out2.mp4")
