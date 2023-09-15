import random
from collections import OrderedDict

# from model.Pose_VQ.VQVAE_pose import VQVAE_2_pose
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchmetrics import F1Score


class Annotation_model(nn.Module):

    def __init__(self, load=True, trainable=False):
        super().__init__()

        self.embedding = nn.Embedding(512, 32)
        self.gru = nn.GRU(input_size=32, hidden_size=16, batch_first=True, bidirectional=True)
        self.norm1 = nn.LayerNorm(16)
        self.annotation_modules = nn.ModuleList()
        self.arcs = nn.ModuleList()
        self.f1s = nn.ModuleList()
        self.classes = [20, 6, 6, 9, 9, 14, 14, 14, 21, 21, 7, 7, 5, 5, 15, 15]

        for i in range(16):
            self.annotation_modules.append(nn.Sequential(
                nn.Flatten(),
                nn.Dropout(0.2),
                nn.Linear(1600, 128),
                nn.Mish(),
                nn.LayerNorm(128),
                nn.Dropout(0.2),
                nn.Linear(128, self.classes[i] * 4),
                nn.Softmax()
            ))
            self.f1s.append(F1Score(task="multiclass", num_classes=self.classes[i]))
        self.occ_net = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(1600, 128),
            nn.Mish(),
            nn.LayerNorm(128),
            nn.Dropout(0.2),
            nn.Linear(128, 4)
        )
        self.anno_strings = ["entity", "l_phase", "r_phase", "l_phrase", "r_phrase", "l_position", "r_position",
                             "s_position", "l_shape", "r_shape", "l_wrist", "r_wrist", "l_extent", "r_extent",
                             "l_practice", "r_practice"]

    def getPoseIDEmbedding(self, ids):
        x = self.embedding(ids)
        x, _ = self.gru(x)
        x = x[:, :, :16] + x[:, :, 16:]
        x = self.norm1(F.mish(x))
        return x

    def getforwardPose(self, pose):
        pose = pose[:, :4, :]
        dec, diff, quant_t, quant_b, id_a, id_b = self.vq.forward_extra(pose.unsqueeze(1))
        ids = torch.cat((id_a.reshape(pose.shape[0], -1), id_b.reshape(pose.shape[0], -1)), dim=-1)
        return ids

    def inference(self, pose_id):
        x = self.getPoseIDEmbedding(pose_id)
        ret = []
        for i in range(16):
            out = self.annotation_modules[i](x).reshape((x.shape[0], 4, -1))
            ret.append(out)
        return torch.cat(ret, dim=-1)

    def forward(self, pose, anno, pose_unsu=None):
        anno = anno.long()
        if random.uniform(0, 1) > 0.5:
            noise = torch.randn(pose.size(), device="cuda") * pose.std() + pose.mean()
            pose = pose + noise / 75
        x = self.getforwardPose(pose)

        if pose_unsu is not None:
            noise = torch.randn(pose_unsu.size(), device="cuda") * pose_unsu.std() + pose_unsu.mean()
            x_unsu_s = self.getforwardPose(F.dropout(pose_unsu + noise / 4, 0.3))

            noise = torch.randn(pose_unsu.size(), device="cuda") * pose_unsu.std() + pose_unsu.mean()
            x_unsu_w = self.getforwardPose(pose_unsu + noise / 10)

        anno1 = torch.cat((anno[:, 0, :].unsqueeze(1), anno[:, 2:, :]), dim=1)

        anno_occ = anno[:, 1, :]
        loss_out = []
        ret_dict = OrderedDict()
        unsu_dif = []
        for i in range(16):
            out = self.annotation_modules[i](x).reshape((x.shape[0], 4, -1))
            l1 = F.cross_entropy(out.reshape((x.shape[0] * 4, -1)), anno1[:, i].reshape((x.shape[0] * 4)))
            loss_out.append(l1)
            a1 = torch.argmax(out, dim=-1)
            ret_dict[self.anno_strings[i]] = self.f1s[i](a1.flatten(), anno1[:, i].flatten())

            if pose_unsu is not None:

                out_unsu_w = self.annotation_modules[i](x_unsu_w).reshape((x_unsu_w.shape[0] * 4, -1))
                a2 = torch.argmax(out_unsu_w, dim=-1)

                out_unsu_s = self.annotation_modules[i](x_unsu_s).reshape((x_unsu_s.shape[0] * 4, -1))

                cond = out_unsu_w.max(dim=1).values
                cond = cond >= 0.9980
                out_unsu_s = out_unsu_s[cond]
                a2 = a2[cond]
                if a2.shape[0] > 0:
                    l1 = F.cross_entropy(out_unsu_s, a2)
                    loss_out.append(l1 / 20)
                    unsu_dif.append(a2.shape[0] / out_unsu_w.shape[0])
        if len(unsu_dif) > 0:
            unsu_dif = np.asarray(unsu_dif).mean()
            ret_dict["unsupervised amount"] = unsu_dif

        l1 = []
        occ_x = self.occ_net(x)
        l_o = F.smooth_l1_loss(occ_x / 0.1, anno_occ / 0.1) * 0.1
        l1.append(l_o)

        l1 = torch.stack(l1).mean()
        ret_dict["occurence"] = l1

        loss_out.append(l1)
        loss = torch.stack(loss_out).mean()

        return loss, ret_dict
