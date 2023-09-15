import math
import random

import scripts.model.embedding_net
import torch
import torch.nn as nn
import torch.nn.functional as F
from scripts.model.VQVAE_pose import VQVAE_2_pose
from scripts.model.annotation_model import Annotation_model
from scripts.model.conc_Dropout import ConcreteDropout
from scripts.model.multimodal_context_net import TextEncoderTCN


class Reshape(nn.Module):
    def __init__(self, *args):
        """
        Reshape layer, using view not reshape for speed reasons.
        @param args:
        """
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


def printParameter(module, name=""):
    disc_total_params = sum(p.numel() for p in module.parameters())
    disc_total_params_train = sum(p.numel() for p in module.parameters() if p.requires_grad)
    print(name + ":", " total parameters:", disc_total_params)
    print(name + ":", " total trainable parameters:", disc_total_params_train)


class PoseEncoder(nn.Module):

    def __init__(self, pose_dim):
        """
        A simple encoding network for the position. Currently only used in the discriminator to better learn the gestures.
        We use Concrete Dropout here, for better accuracy (https://github.com/danielkelshaw/ConcreteDropout)
        @param pose_dim: sequence length with pre gesture frames (34 in our case).
        """
        super().__init__()
        self.input_size = (pose_dim, 159)
        self.pose_encoder = nn.Sequential(
            nn.Conv1d(self.input_size[0], out_channels=16, kernel_size=(3,), padding=1, bias=False),
            nn.BatchNorm1d(16),
            nn.Mish(True),
            nn.MaxPool1d(2),
            nn.Conv1d(16, out_channels=16, kernel_size=(3,), padding=1, bias=False),
            nn.BatchNorm1d(16),
            nn.Mish(True),
            nn.MaxPool1d(2),
            nn.Conv1d(16, out_channels=32, kernel_size=(3,), padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.Mish(True),
            nn.MaxPool1d(2),
            nn.Conv1d(32, out_channels=32, kernel_size=(3,), padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.Mish(True),
            nn.MaxPool1d(2),
            nn.Conv1d(32, out_channels=64, kernel_size=(3,), padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.Mish(True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, out_channels=64, kernel_size=(3,), padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.Flatten(),
            nn.Mish(True),
        )
        self.dropout_encoder = ConcreteDropout()

    def forward(self, input):
        return self.dropout_encoder(input, self.pose_encoder)


class EmbeddingNet(nn.Module):

    def __init__(self, classes, z_size, hidden_size=None):
        """
        The Embedding network. We use this network to learn an embedding for each annotation, without handcoding each one.
        @param classes: number of possuble classes
        @param z_size: output dimension size. Has to be declared, as we an also only use the network for inference, without annotations.
        @param hidden_size: the internal hidden dimension size
        """
        super().__init__()

        if hidden_size is None:
            hidden_size = z_size
        self.z_size = z_size

        self.embedding = nn.Sequential(
            nn.Embedding(classes, hidden_size),  # 19 possible entities
            nn.Linear(hidden_size, hidden_size),
            nn.Mish(True),
        )
        self.embedding2 = nn.Sequential(
            nn.Linear(hidden_size, z_size),
            nn.Mish(True),
        )
        self.mu = nn.Linear(z_size, z_size)
        self.logvar = nn.Linear(z_size, z_size)

    def forward(self, data):
        """
        The forward function. This takes the annotation data and only trains the embedding if there are
        annotations present (id > -1). Additionally, we ignore the annotation 20 percent of the time to make the
        training more robust. If there are no annotations, the ret data is zero. During inference this embedding can be
        used to modulate the generated gestures.
        @param data: annotation input
        @return: embedding vector
        """
        ret = data.new_zeros((data.shape[0], self.z_size), dtype=torch.float32)
        idx = data.flatten() > -1
        if idx.nonzero().shape[0] > 0 and (not self.training or random.uniform(0, 1) >= 0.2):
            z_context = self.embedding(data[idx.nonzero()])
            z_context = self.embedding2(F.dropout(z_context, 0.2, self.training))
            z_mu = self.mu(z_context)
            z_logvar = self.logvar(z_context)
            z_context = scripts.model.embedding_net.reparameterize(z_mu, z_logvar)
            ret[idx.nonzero()] += z_context

        return ret


class AqgtPlusGenerator(nn.Module):

    def __init__(self, args, pose_dim, n_words, word_embed_size, word_embeddings, z_obj=None):
        """
        The initializer of our Generator Network.
        @param args: the config arguments
        @param pose_dim: the pose dimension of our gestures
        @param n_words: number of words for the text encoder
        @param word_embed_size: word embedding size for the text encoder
        @param word_embeddings: word embeddings of the text encoder
        @param z_obj: boolean if we want to learn a speaker identity embedding. Mainly used for testing purposes.
        """
        super().__init__()

        self.args = args
        self.dropout = args.dropout_prob
        self.pre_length = args.n_pre_poses
        self.gen_length = args.n_poses - args.n_pre_poses
        self.z_obj = z_obj
        self.input_context = args.input_context

        if self.input_context == 'both':
            self.in_size = 32 + 32 + pose_dim + 1  # audio_feat + text_feat + last pose + constraint bit
        elif self.input_context == 'none':
            self.in_size = pose_dim + 1
        else:
            self.in_size = 32 + pose_dim + 1  # audio or text only

        self.time_step = 34
        self.beat_size = 10 * 3 * 3
        self.side_size = 10 * 3
        self.hidden_size = args.hidden_size

        # audio encoding networks
        self.audio_bottom1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2304, self.hidden_size),
            nn.Mish(True),
        )

        self.audio_bottom2 = nn.Sequential(
            nn.Linear(self.hidden_size, 64 * args.n_poses),
            nn.Mish(True),
        )

        self.audio_top1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(576, self.hidden_size),
            nn.Mish(True),
        )

        self.audio_top2 = nn.Sequential(
            nn.Linear(self.hidden_size, 64 * args.n_poses),
            nn.Mish(True),
        )

        self.audio_norm = nn.LayerNorm(64)

        # context encoding networks
        if self.z_obj:
            self.z_size = 16
            self.in_size += self.z_size
            print("making speaker embedding model")
            self.speaker_embedding1 = nn.Sequential(
                nn.Embedding(3329, self.z_size),
                nn.Linear(self.z_size, self.z_size),
                nn.Mish(True),
            )
            self.speaker_embedding2 = nn.Sequential(
                nn.Linear(self.z_size, self.z_size),
            )
            self.speaker_mu = nn.Linear(self.z_size, self.z_size)
            self.speaker_logvar = nn.Linear(self.z_size, self.z_size)

        self.z_arc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(34 * self.hidden_size, self.hidden_size),
            nn.Mish(True),
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, 3329),
            nn.Softmax(),
        )

        self.z_emb = nn.Embedding(3329, self.z_size)


        # GRU-Transformer layer
        self.dummy_param = nn.Parameter(torch.empty(0))
        self.input_gru = 1064
        self.gru_layer_norm = nn.LayerNorm(self.input_gru)

        self.transformer_1 = Simple_Transformer(self.input_gru,  # prepose encoding
                                                hidden_size=self.hidden_size,
                                                num_layers=2)
        self.transformer_2 = Simple_Transformer(self.hidden_size,  # output of first encoder
                                                hidden_size=self.hidden_size,
                                                num_layers=5,
                                                dropout=self.dropout)
        self.gru_1 = nn.GRU(self.input_gru,
                            hidden_size=self.hidden_size,
                            num_layers=1,
                            dropout=self.dropout, batch_first=True, bidirectional=True)
        self.gru_2 = nn.GRU(self.hidden_size * 2,  # output of first encoder
                            hidden_size=self.hidden_size,
                            num_layers=4,
                            dropout=self.dropout, batch_first=True, bidirectional=True)

        # Text encoder
        self.text_encoder = TextEncoderTCN(args, n_words, word_embed_size, pre_trained_embedding=word_embeddings,
                                           dropout=self.dropout)


        # beat encoding, adapted from the SEEG paper: https://github.com/akira-l/SEEG
        self.beat_emb1 = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.Mish(True),
        )
        self.beat_emb2 = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.Mish(True),
        )

        self.pose_emb1 = nn.Sequential(
            nn.Linear(160, self.hidden_size),
            nn.Mish(True),
        )

        self.pose_emb2 = nn.Sequential(
            nn.Linear(self.hidden_size, 64),
            nn.Mish(True),
        )

        self.gru_out_norm = nn.LayerNorm(2224)

        self.out_emb1 = nn.Sequential(
            nn.Linear(2224, self.hidden_size * 2),
            nn.Mish(True),
        )

        self.out_emb2 = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.Mish(True),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Mish(True),
        )

        self.beat_out_emb = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.Mish(True),
            nn.Linear(self.hidden_size, 159),
        )

        self.square_layer1 = nn.Sequential(
            nn.Linear(3 * self.side_size, self.hidden_size),
            nn.Mish(True),
        )

        self.square_layer2 = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Mish(True),
        )

        self.oenv_layer1 = nn.Sequential(
            nn.Linear(3 * self.side_size, self.hidden_size),
            nn.Mish(True),
        )

        self.oenv_layer2 = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Mish(True),
        )

        # temporal aligner networks

        self.temporal_align_norm = nn.LayerNorm(159 * 5 + 16)

        self.temporal_align1 = nn.Sequential(
            nn.Linear(159 * 5 + 16, self.hidden_size),
            nn.Mish(True),
        )

        self.temporal_align2 = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Mish(True),
        )

        self.temporal_align3 = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
        )

        self.temporal_align_out_norm = nn.LayerNorm(659)

        self.temporal_align4 = nn.Sequential(
            nn.Linear(659, 400),
        )

        self.temporal_align5 = nn.Sequential(
            nn.Linear(400, 159),
            nn.Tanh()
        )

        # pretrained position encoder
        self.seq_pose_encoder = VQVAE_2_pose(in_channel=1, channel=256, embed_dim=4, n_res_block=4, n_res_channel=128)
        dict = torch.load("pretrained/pose_vq/pose_vq.bin")
        self.seq_pose_encoder.load_state_dict(dict["gen_dict"])
        self.seq_pose_encoder.eval()
        for p in self.seq_pose_encoder.parameters():
            p.requires_grad = False
        self.pose_embeddings = nn.Sequential(
            nn.Embedding(512, 128),
            nn.GRU(128, hidden_size=17, num_layers=1, batch_first=True, bidirectional=True)
        )

        # pretrained annotation network
        self.anno_net = Annotation_model()
        dict = torch.load("pretrained/annotation_model/annotation_model.bin")
        self.anno_net.load_state_dict(dict["gen_dict"], strict=False)
        self.anno_net.eval()
        for p in self.anno_net.parameters():
            p.requires_grad = False
        self.anno_GRU = nn.GRU(188, hidden_size=16, num_layers=1, batch_first=True, bidirectional=True)

        # Wav2Vec2 output encoder
        self.Wav2Vec2modelGRU = nn.GRU(113, hidden_size=17, num_layers=1, batch_first=True, bidirectional=True)
        self.Wav2Vec2model_norm = nn.LayerNorm(34)

        self.do_flatten_parameters = False
        if torch.cuda.device_count() > 1:
            self.do_flatten_parameters = True

        self.latent_loss_weight = 0.25
        self.vid_max = 0

        # We create an embedding network for each dimensions of our annotations.
        anno_sizes = [20, 1, 6, 6, 9, 9, 14, 14, 14, 21, 21, 7, 7, 6, 6, 15, 15]
        self.anno_emb = nn.ModuleList()
        for a_s in anno_sizes:
            if a_s == 1:
                self.anno_emb.append(nn.Sequential(
                    nn.Linear(1, self.z_size),
                    nn.Mish(True),
                ))
            else:
                self.anno_emb.append(EmbeddingNet(a_s, self.z_size))

        self.emb_condenser = nn.Sequential(
            nn.Linear(self.z_size * len(anno_sizes), self.z_size * 2),
            nn.Mish(),
            nn.Linear(self.z_size * 2, self.z_size * 2)
        )

    def toMaps(self, annotation):
        ret = []
        for i in range(17):
            ret.append(annotation[:, i, :])
        return ret

    def forward(self, pre_seq, beats, in_text, in_audio, audio_var_bottom, audio_var_top, vid_indices, annotation):
        """
        The forward function of our generator model, which combines the forward pass for the Generator and the Temporal Aligner.
        @param pre_seq: The first frames of the previous gestures
        @param beats: SEEG beat information
        @param in_text: Text input
        @param in_audio: Audio input preprocessed with Wav2Vec2
        @param audio_var_bottom: VQVAE bottom input
        @param audio_var_top: VQVAE top input
        @param vid_indices: speaker indices input
        @param annotation: annotation input
        @return: A new gesture sequence
        """
        batch_size = pre_seq.shape[0]
        # we convert the given annotations to an array format, for better lookup
        annotation = self.toMaps(annotation)

        # context encoding for the speaker identity. Taken from the Trimodal Context paper.
        z_context = self.speaker_embedding1(vid_indices)
        z_context = self.speaker_embedding2(F.dropout(z_context, self.dropout, self.training))
        z_mu = self.speaker_mu(z_context)
        z_logvar = self.speaker_logvar(z_context)
        z_context = scripts.model.embedding_net.reparameterize(z_mu, z_logvar)

        # pre pose encoding using the sequence encoder.
        pose = pre_seq[:, :4, :159]
        dec, diff, quant_t, quant_b, id_a, id_b = self.seq_pose_encoder.forward_extra(pose.unsqueeze(1))
        ids = torch.cat((id_a.reshape(pose.shape[0], -1), id_b.reshape(pose.shape[0], -1)), dim=-1)
        quant_pose, _ = self.pose_embeddings(ids)
        quant_pose = quant_pose.permute(0, 2, 1)

        # Annotation encoding for predicting the annotation information of the first four frames.
        # This helps in stabilising the inference and the gesture generation if no annotations are present.
        anno_cat = self.anno_net.inference(ids)
        anno_cat, _ = self.anno_GRU(anno_cat)
        anno_cat = anno_cat.reshape(pre_seq.shape[0], 1, -1).repeat(1, 34, 1)

        # text encoding using the text encoder
        text_feat_seq = self.text_encoder(in_text)
        text_feat_seq = text_feat_seq.repeat(1, 34, 1)

        # audio encoding. Both for the top and bottom input of vqvae input
        audio_var_bottom = self.audio_bottom1(F.dropout(audio_var_bottom, self.dropout, self.training))
        audio_var_bottom = self.audio_bottom2(audio_var_bottom)
        audio_var_top = self.audio_top1(F.dropout(audio_var_top, self.dropout, self.training))
        audio_var_top = self.audio_top2(audio_var_top)
        audio_comb = audio_var_bottom + audio_var_top
        audio_comb = audio_comb.reshape((audio_comb.shape[0], self.time_step, 64))
        audio_comb = self.audio_norm(F.mish(audio_comb))

        # second audio encoding for the Wav2vec input
        log, _ = self.Wav2Vec2modelGRU(in_audio.squeeze(1).permute(0, 2, 1))
        audio_log = self.Wav2Vec2model_norm(F.mish(log)).permute(0, 2, 1)

        # Annotation encoding for the created annotation array.
        # For every category we learn an embedding using the EmbeddingNet.
        c1 = []
        for i, anno in enumerate(annotation):
            anno_f = anno.reshape(batch_size * self.time_step)
            out = self.anno_emb[i](anno_f.long() if i != 1 else anno_f.unsqueeze(1))
            out = out.reshape((batch_size, self.time_step, self.z_size))
            c1.append(out)
        c1 = torch.cat(c1, dim=2)
        anno_z = self.emb_condenser(F.dropout(c1, self.dropout, self.training))

        # The beat calculation, adapted from the SEEG paper.
        sub_b = []
        bs, _, _ = beats.size()
        for ts in range(self.time_step):

            if ts == 0:
                sub_beats = torch.cat([torch.zeros(bs, 2, self.side_size, device=self.dummy_param.device),
                                       beats[:, :, :2 * self.side_size]], 2)

            elif ts == self.time_step - 1:
                sub_beats = torch.cat([beats[:, :, -2 * self.side_size:],
                                       torch.zeros(bs, 2, self.side_size, device=self.dummy_param.device)], 2)
            else:
                sub_beats = beats[:, :, self.side_size * (ts - 1):self.side_size * (ts + 2)]

            sub_b.append(sub_beats.unsqueeze(1))
        sub_b = torch.cat(sub_b, dim=1)
        squ_out = self.square_layer1(F.dropout(sub_b[:, :, 0, :], self.dropout, self.training))
        squ_out = self.square_layer2(F.dropout(squ_out, self.dropout, self.training))

        oenv_out = self.oenv_layer1(F.dropout(sub_b[:, :, 1, :], self.dropout, self.training))
        oenv_out = self.oenv_layer2(F.dropout(oenv_out, self.dropout, self.training))
        sum_out = squ_out + oenv_out

        pose_out = self.pose_emb1(F.dropout(pre_seq, self.dropout, self.training))
        pose_out = self.pose_emb2(F.dropout(pose_out, self.dropout, self.training))

        # We combine all previous encodings to learn the GRU-Transformer network with it.
        cat_out = torch.cat([pose_out, sum_out, audio_comb, audio_log, text_feat_seq, quant_pose,
                             z_context.unsqueeze(1).repeat(1, 34, 1), anno_z, anno_cat], 2)


        # The GRU transformer


        # First Transformer
        cat_out = self.gru_layer_norm(cat_out)
        output = self.transformer_1(cat_out)
        output_1 = F.mish(output)

        # First GRU
        output_gru, h = self.gru_1(cat_out)
        output_gru = output_gru[:, :, :self.hidden_size] + output_gru[:, :, self.hidden_size:]
        output_gru = F.mish(output_gru)

        # Second Transformer
        output = self.transformer_2(F.dropout(output_1, self.dropout, self.training))
        output_2 = F.mish(output)

        # Second GRU
        output, _ = self.gru_2(F.dropout(torch.cat((output_gru, output_2), dim=2), self.dropout, self.training))
        output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]
        output = F.mish(output)

        # The second networks give a small speedup by reducing the internal hidden state and allow for interconnection
        # between the two architectues. This could be adapted to more layers and more width, but this becomes increasingly
        # slow.


        # Combination of all outputs, as well as the skip connections of the earlier text and audio features.
        ret_anno = {}
        output_fin = torch.cat([output, output_1, output_2, output_gru, audio_comb, audio_log, text_feat_seq], 2)
        output_fin = self.gru_out_norm(F.dropout(output_fin, self.dropout, self.training))
        feat_out = self.out_emb1(F.dropout(output_fin, self.dropout, self.training))
        feat_out = self.out_emb2(F.dropout(feat_out, self.dropout, self.training))

        # Forward pass of the speaker decision network. This network uses the previous information to learn a softmax
        # classification of the possible speaker. As the model already knows the correct information from the first context encoding,
        # this mainly functions to ensure that the model adheres to the speaker identity.
        arc_out = self.z_arc(F.dropout(feat_out, self.dropout, self.training))
        arc_loss = F.cross_entropy(arc_out, vid_indices)
        arc_out = self.z_emb(torch.argmax(arc_out, dim=1)).unsqueeze(1).repeat(1, 34, 1)

        # The Temporal Aligner

        # We define two lists, one for the sequence output vg and one for the "p" skip connection vector.
        ret = []
        ret2 = []
        last_feat = pre_seq.new_zeros((pre_seq.shape[0], self.hidden_size))

        # We add zero vectors for easier calculation
        for i in range(4):
            ret.append(pre_seq.new_zeros((pre_seq.shape[0], 4, 159)))

        # For every step, we take the four previous outputs and the current frame to align the gestures.
        # We give these information to the vqvae decoder, to generate a new sequence.
        for ts in range(self.time_step):
            fout = feat_out[:, ts, :]
            beat_fin_out = self.beat_out_emb(F.dropout(F.mish(torch.cat((fout, last_feat), dim=1)), self.dropout, self.training))
            last_feat = fout

            g1 = torch.cat((ret[ts - 4][:, 1], ret[ts - 3][:, 1], ret[ts - 2][:, 1], ret[ts - 1][:, 1], beat_fin_out,
                            arc_out[:, ts, :]), dim=-1)
            g1 = self.temporal_align_norm(g1)
            tt_1 = self.temporal_align1(F.dropout(F.mish(g1), self.dropout, self.training))
            tt_1 = self.temporal_align2(F.dropout(tt_1, self.dropout, self.training))
            tt_1 = self.temporal_align3(F.dropout(tt_1, self.dropout, self.training))

            g2 = self.temporal_align_out_norm(torch.cat((tt_1, beat_fin_out.squeeze(1)), dim=1))

            tt_1 = self.temporal_align4(F.dropout(F.mish(g2), self.dropout, self.training))
            quant_b = tt_1[:, :320].reshape((tt_1.shape[0], 4, 2, 40))
            quant_t = tt_1[:, 320:].reshape((tt_1.shape[0], 4, 1, 20))
            dec = self.seq_pose_encoder.decode(quant_t, quant_b)
            dec = dec[:, 0, :4, :159]
            ret.append(dec)
            ret2.append(self.temporal_align5(tt_1))

        # Finally we construct the output vector by combining the gesture vector g out of the third element of vg_−1, the second element of vg, and the first element of vg_+1.
        out = []
        ret = ret[4:]
        for i in range(self.time_step):
            if i > 0 and i < self.time_step - 1:
                out.append(((ret[i - 1][:, 2, :].unsqueeze(1) + ret[i][:, 1, :].unsqueeze(1) + ret[i + 1][:, 0,
                                                                                               :].unsqueeze(1)) / 3) +
                           ret2[i].unsqueeze(1))
            elif i == 0:
                out.append(
                    ((ret[i][:, 1, :].unsqueeze(1) + ret[i + 1][:, 0, :].unsqueeze(1)) / 2) + ret2[i].unsqueeze(1))
            else:
                out.append(
                    ((ret[i - 1][:, 2, :].unsqueeze(1) + ret[i][:, 1, :].unsqueeze(1)) / 2) + ret2[i].unsqueeze(1))

        out = torch.cat(out,dim=1)

        # we return the context vectors, the arc loss (which is the auxilliary loss for all embeddings.
        # Ret_anno is returned, but is currently not used in the final network.
        return out, z_context, z_mu, z_logvar, arc_loss, ret_anno


class MyLoopDiscriminator(nn.Module):
    def __init__(self, input_size, args, n_words, word_embed_size, word_embeddings):
        """
        Initialization for our discriminator. We combine every modality and feed the output through a GRU layer.
        @param input_size: Input size to modulate size of all layers. Currently unused.
        @param args: config args file
        @param n_words: Number of possible words, used for the fasttext textencoder
        @param word_embed_size: Word embedding size. Used for the textencoder
        @param word_embeddings: The actual embeddings used for the textencoder.
        """
        super().__init__()
        self.input_size = input_size

        self.dropout = args.dropout_prob

        self.hidden_size = args.hidden_size
        self.time_step = 34
        self.side_size = 1

        self.bnorm = nn.LayerNorm(319)
        self.gru = nn.GRU(319, hidden_size=32, num_layers=2,
                          bidirectional=True,
                          dropout=self.dropout, batch_first=True)
        self.anorm = nn.LayerNorm(32)

        self.fnorm = nn.LayerNorm(256 + 32 * 34)
        self.fin_out1 = nn.Sequential(
            nn.Linear(256 + 32 * 34, self.hidden_size),
            nn.Mish(True),
        )
        self.fin_norm = nn.LayerNorm(self.hidden_size)
        self.fin_out2 = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Mish(True),
        )
        self.fin_norm2 = nn.LayerNorm(self.hidden_size)
        self.fin_out3 = nn.Sequential(
            nn.Linear(self.hidden_size, 1)
        )

        self.audio_bottom1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2304, self.hidden_size),
            nn.Mish(True),
        )

        self.audio_bottom2 = nn.Sequential(
            nn.Linear(self.hidden_size, 32 * args.n_poses),
            nn.Mish(True),
        )

        self.audio_top1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(576, self.hidden_size),
            nn.Mish(True),
        )

        self.audio_top2 = nn.Sequential(
            nn.Linear(self.hidden_size, 32 * args.n_poses),
            nn.Mish(True),
        )

        self.anno_net = nn.Sequential(
            nn.Linear(34 * 17, self.hidden_size),
            nn.Mish(True),
            nn.Linear(self.hidden_size, self.hidden_size)
        )

        self.pose_encoder = PoseEncoder(34)

        self.text_encoder = TextEncoderTCN(args, n_words, word_embed_size, pre_trained_embedding=word_embeddings,
                                           dropout=args.dropout_prob)

        self.do_flatten_parameters = False
        if torch.cuda.device_count() > 1:
            self.do_flatten_parameters = True

    def forward(self, poses, in_text, in_audio, audio_var_bottom, audio_var_top, annotation):
        """
        forward function of our discriminator. The model combines all inputs and feeds them through a GRU.
        As cudnn doesn't allow for double backwards passes with GRU layers we temporarily disable cudnn.
        In the future this should be changed, as this makes training around 10% slower.

        @param poses: The gesture input
        @param in_text: The text input
        @param in_audio: Currently unused as the audio_var_bottom and top already contain the audio information
        @param audio_var_bottom: Output of the audio VQVAE (bottom encoder)
        @param audio_var_top: Output of the audio VQVAE (top encoder)
        @param annotation: The annotation information
        @return: vector of size [batchsize,1] for the gan training
        """
        audio_var_bottom = self.audio_bottom1(F.dropout(audio_var_bottom, self.dropout, self.training))
        audio_var_bottom = self.audio_bottom2(F.dropout(audio_var_bottom, self.dropout, self.training))

        audio_var_top = self.audio_top1(F.dropout(audio_var_top, self.dropout, self.training))
        audio_var_top = self.audio_top2(F.dropout(audio_var_top, self.dropout, self.training))

        audio_comb = audio_var_bottom + audio_var_top
        audio_comb = audio_comb.reshape((audio_comb.shape[0], self.time_step, 32))

        text_feat_seq = self.text_encoder(in_text)

        pose_enc = self.pose_encoder(poses)

        input_data = torch.cat((poses, audio_comb, text_feat_seq.repeat(1, 34, 1)), dim=2)
        input_data = self.bnorm(F.mish(input_data))

        with torch.backends.cudnn.flags(enabled=False):
            output, decoder_hidden = self.gru(input_data)
        output = output[:, :, :32] + output[:, :, 32:]
        output = self.anorm(output)
        output = F.mish(output)

        output = output.reshape((-1, 32 * 34))
        output = torch.cat((pose_enc, output), dim=1)
        output = self.fnorm(output)
        fin_out = self.fin_out1(F.dropout(output, self.dropout, self.training))

        if annotation is not None:
            addon_data = self.anno_net(annotation.reshape((annotation.shape[0], -1)).float())
            fin_out = fin_out + addon_data
        fin_out = self.fin_norm(fin_out)
        fin_out = self.fin_out2(F.dropout(fin_out, self.dropout, self.training))

        fin_out = self.fin_norm2(fin_out)
        fin_out = self.fin_out3(fin_out)

        return fin_out


class Simple_Transformer(nn.Module):

    def __init__(self, in_features, hidden_size, num_layers, max_heads=8, dropout=0.2):
        """
        Simple Transformer adapted from the "Attention is all you need" paper implementation.
        @param in_features: Number of features
        @param hidden_size: Number of hidden vectors
        @param num_layers: How many layers should be created internally
        @param max_heads: Number of attention heads. They need to be dividable by the number of features
        @param dropout: Should dropout be applied and if yes, how much.
        """
        super().__init__()

        self.pos_encoder = nn.ModuleList()
        self.transformers = nn.ModuleList()

        self.layer_norms = nn.ModuleList()

        h = in_features
        h_new = h
        self.num_layers = num_layers
        for i in range(num_layers):
            h_new = max(h // 2, hidden_size)
            self.pos_encoder.append(PositionalEncoder(h, max_seq_len=34))
            self.transformers.append(nn.Sequential(
                nn.TransformerEncoderLayer(h, getDivider(h, max_heads=max_heads), dropout=dropout, batch_first=True),
                nn.Mish(),
                nn.Dropout(dropout),
                nn.Linear(h, h_new),
                nn.Mish(),
                ))
            self.layer_norms.append(nn.LayerNorm(h_new))
            h = h_new
        self.out_net = nn.Sequential(nn.Dropout(dropout), nn.Linear(h_new, hidden_size))

    def forward(self, x):
        """
        forward function. Contrary to the original paper we add positional encoding for each layer.
        For some reason this slightly improves performance.
        @param x: input
        @return: output
        """
        for i in range(self.num_layers):
            x = self.pos_encoder[i](x)
            x = self.transformers[i](x)
            x = self.layer_norms[i](x)
        return self.out_net(x)


class PositionalEncoder(torch.nn.Module):
    def __init__(self, d_model, max_seq_len=160):
        """
        Positional Encoding taken from "Attention is all you need" paper implementation
        @param d_model:
        @param max_seq_len:
        """
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model - 1, 2):
                pe[pos, i] = \
                    math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = \
                    math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x * math.sqrt(self.d_model)
        seq_len = x.size(1)
        pe = self.pe[:, :seq_len]
        x = x + pe
        return x


def getDivider(dimension, max_heads=12):
    """
    Small function to calculate the biggest dividable attention head number.
    @param dimension: dimension of the input.
    @param max_heads: maximum number of heads that should be returned.
    @return: number of heads
    """
    mret = 1
    for i in range(1, max_heads + 1):
        if (dimension / i).is_integer():
            mret = i
    #print("divider is:", mret)
    return mret


if __name__ == '__main__':
    pp = PoseEncoder(34)
