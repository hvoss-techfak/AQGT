import logging
import math
import os
import random
import subprocess
import time
import traceback
from logging.handlers import RotatingFileHandler
from textwrap import wrap

import cv2
import matplotlib
import numpy as np
import soundfile as sf
from tqdm import tqdm

from scripts.utils.SkeletonHelper.VisualizeHelper import VisualizeCvClass
from scripts.utils.data_utils import skeleton_parents

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import matplotlib.animation as animation

import scripts.utils.data_utils
import scripts.data_loader.lmdb_data_loader


# only for unicode characters, you may remove these two lines
from scripts.model import vocab

matplotlib.rcParams['axes.unicode_minus'] = False


def set_logger(log_path=None, log_filename='log'):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    handlers = [logging.StreamHandler()]
    if log_path is not None:
        os.makedirs(log_path, exist_ok=True)
        handlers.append(
            RotatingFileHandler(os.path.join(log_path, log_filename), maxBytes=10 * 1024 * 1024, backupCount=5))
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s: %(message)s', handlers=handlers)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since):
    now = time.time()
    s = now - since
    return '%s' % as_minutes(s)

def create_video_and_save_single(save_path, vid, sent, target, mean_data, title,
                          audio=None, aux_str=None, clipping_to_shortest_stream=False, delete_audio_file=True, save_flag=False):
    print('rendering a video...')
    start = time.time()

    fig = plt.figure(figsize=(4, 4))

    #axe = plt.axes() 
    axe = plt.axes(projection='3d') 

    #axes = [fig.add_subplot(1, 2, 1, projection='3d'), fig.add_subplot(1, 2, 2, projection='3d')]
    #axes[0].view_init(elev=20, azim=-60)
    #axes[1].view_init(elev=20, azim=-60)
    fig_title = title

    if aux_str:
        fig_title += ('\n' + aux_str)
    fig.suptitle('\n'.join(wrap(fig_title, 75)), fontsize='medium')

    # un-normalization and convert to poses
    mean_data = mean_data.flatten()
    #output = output + mean_data
    #output_poses = utils.data_utils.convert_dir_vec_to_pose(output)
    target_poses = None
    if target is not None:
        target = target + mean_data
        target_poses = scripts.utils.data_utils.convert_dir_vec_to_pose(target)

    def animate(i):
        pose = target_poses[i]

        if pose is not None:
            axe.clear()
            for j, pair in enumerate(scripts.utils.data_utils.dir_vec_pairs):

                axe.get_xaxis().set_visible(False) 
                axe.get_yaxis().set_visible(False) 
                axe.get_zaxis().set_visible(False) 
                axe.grid(False) 
                plt.axis('off') 

                axe.plot([pose[pair[0], 0], pose[pair[1], 0]],
                              [pose[pair[0], 2], pose[pair[1], 2]],
                              [pose[pair[0], 1], pose[pair[1], 1]],
                             zdir='z', linewidth=8)
                #axe.plot(pose[pair[0], 0],
                #             pose[pair[0], 2],
                #             pose[pair[0], 1],
                #             zdir='z', linewidth=5)
                axe.set_xlim3d(-0.5, 0.5)
                axe.set_ylim3d(0.5, -0.5)
                axe.set_zlim3d(0.5, -0.5)
                #axe.set_xlabel('x')
                #axe.set_ylabel('z')
                #axe.set_zlabel('y')
                axe.set_title('{} ({}/{})'.format('gt', i + 1, len(target)))

    def save_figure(save_path, vid_name):
        #pose = target_poses[i]
        #if pose is not None:

        fig_save_dir = os.path.join(save_path, 'fig-{}'.format(vid_name)) 
        os.makedirs(fig_save_dir,exist_ok=True)

        for cnt, pose in enumerate(target_poses): 
            save_fig = plt.figure(figsize=(4, 4))
            save_axe = plt.axes(projection='3d') 
            save_axe.clear()

            #save_fig.suptitle('\n'.join(wrap(fig_title, 75)), fontsize='medium')

            for j, pair in enumerate(scripts.utils.data_utils.dir_vec_pairs):

                save_axe.get_xaxis().set_visible(False) 
                save_axe.get_yaxis().set_visible(False) 
                save_axe.get_zaxis().set_visible(False) 
                save_axe.grid(False) 
                plt.axis('off') 

                save_axe.plot([pose[pair[0], 0], pose[pair[1], 0]],
                              [pose[pair[0], 2], pose[pair[1], 2]],
                              [pose[pair[0], 1], pose[pair[1], 1]],
                             zdir='z', linewidth=8)
                #axe.plot(pose[pair[0], 0],
                #             pose[pair[0], 2],
                #             pose[pair[0], 1],
                #             zdir='z', linewidth=5)
                save_axe.set_xlim3d(-0.5, 0.5)
                save_axe.set_ylim3d(0.5, -0.5)
                save_axe.set_zlim3d(0.5, -0.5)
                #axe.set_xlabel('x')
                #axe.set_ylabel('z')
                #axe.set_zlabel('y')
                #save_axe.set_title('{} ({}/{})'.format('gt', cnt + 1, len(target)))

            fig_save_path = os.path.join(fig_save_dir,  'save_fig' + str(cnt) + '.png') 
            plt.savefig( fig_save_path, bbox_inches='tight', 
                    dpi=save_fig.dpi, pad_inches=0.0)
            plt.clf() 



    num_frames = len(target) 
    ani = animation.FuncAnimation(fig, animate, interval=30, frames=num_frames, repeat=False)
    resave = save_figure(save_path, vid)  


    if save_flag: 
        # show audio
        audio_path = None
        if audio is not None:
            assert len(audio.shape) == 1  # 1-channel, raw signal
            audio = audio.astype(np.float32)
            sr = 16000
            audio_path = '{}/{}_{}.wav'.format(save_path, vid, sent)
            sf.write(audio_path, audio, sr)
    
        # save video
        try:
            video_path = '{}/temp_{}.mp4'.format(save_path, vid)
            ani.save(video_path, fps=15, dpi=80)  # dpi 150 for a higher resolution
            del ani
            plt.close(fig)
        except RuntimeError:
            assert False, 'RuntimeError'
    
        # merge audio and video
        if audio is not None:
            merged_video_path = '{}/{}.mp4'.format(save_path, vid)
            cmd = ['ffmpeg', '-loglevel', 'panic', '-y', '-i', video_path, '-i', audio_path, '-strict', '-2',
                   merged_video_path]
            if clipping_to_shortest_stream:
                cmd.insert(len(cmd) - 1, '-shortest')
            subprocess.call(cmd)
            if delete_audio_file:
                os.remove(audio_path)
            os.remove(video_path)

    print('done, took {:.1f} seconds'.format(time.time() - start))
    #return output_poses, target_poses
    return None 



def create_video_and_save(save_path, epoch, prefix, iter_idx, target, output, mean_data, title,
                          audio=None, aux_str=None,anno=None,withAnnotations=True,save_gen=False, clipping_to_shortest_stream=False, delete_audio_file=True,video_filename="out"):
    print('rendering a video...')


    start = time.time()
    import moviepy.editor as mpy

    fig = plt.figure(figsize=(16, 9))
    axes = [fig.add_subplot(1, 2, 1, projection='3d'), fig.add_subplot(1, 2, 2, projection='3d')]
    axes[0].view_init(elev=20, azim=-60)
    axes[1].view_init(elev=20, azim=-60)
    fig_title = title



    if aux_str:
        fig_title += ('\n' + aux_str)
    fig.suptitle('\n'.join(wrap(fig_title, 75)), fontsize='medium')

    # un-normalization and convert to poses
    mean_data = mean_data.reshape((53,3))
    print(mean_data.min(),mean_data.max(),np.std(mean_data),np.mean(mean_data),np.median(mean_data))
    output = output.reshape((-1,53,3)) + mean_data
    output_poses = scripts.utils.data_utils.convert_dir_vec_to_pose(output)
    target_poses = target.copy().reshape((-1,53,3))
    if target is not None:
        #target_poses = target
        #print(mean_data)
        target_poses = target_poses + mean_data
        #for i in range(34):
        #    target_poses[i] = (target[i] + mean_data.reshape(53,3))
        target_poses = scripts.utils.data_utils.convert_dir_vec_to_pose(target_poses)



    target_poses = target_poses.reshape(-1,159)
    for i in range(1,target_poses.shape[0]-1):
        target_poses[i] = (target_poses[i-1] + target_poses[i] + target_poses[i+1])/3

    if target_poses.shape[-1] != 3:
        target_poses = target_poses.reshape((-1,53,3))
    vis = VisualizeCvClass()

    frame_out = []

    video_path = '{}/temp_{}_{:03d}_{}.mp4'.format(save_path, prefix, epoch, iter_idx)
    output_imgs = []
    target_imgs = []
    comb_imgs = []

    def inv_dict(dic):
        inv_map = {v+1: k for k, v in dic.items()}
        return inv_map

    entity_list = inv_dict({"Hufeisen": 0, "Stockwerk": 1, "Treppe": 2, "Fenster": 3, "Kirche": 4, "Tür": 5, "Aussenobjekte": 6, "Lampen": 7, "Turm": 8, "Schale": 9, "Straße": 10, "Rathaus": 11, "Platz": 12, "Gebäude": 13, "Kunstobjekt": 14, "Dach": 15, "Brunnen": 16, "Hecke": 17})
    phase_list = inv_dict({"prep": 0, "stroke": 1, "retr": 2, "post.hold": 3, "pre.hold": 4})
    phrase_list = inv_dict({"beat": 0, "iconic": 1, "deictic": 2, "iconic-deictic": 3, "discourse": 4, "move": 5, "iconic-deictic-beat": 7})
    position_list = inv_dict({'links': 0, 'drauf': 1, 'rechts': 2, 'davor': 3, 'drinnen': 4, 'hier': 5, 'daneben': 6, 'dahinter': 7, 'zusammen': 8, 'drunter': 9, 'hin': 10, 'drumherum': 11, 'zwischen': 12})
    shape_list = inv_dict({'B_spread': 0, 'B_loose_spread': 1, 'G': 2, '5_bent': 3, 'C': 4, 'B_spread_loose': 1, 'G_loose': 5, '5': 6, 'C_loose': 7, 'G_bent': 8, 'B': 9, 'C_large': 10, 'B_loose': 11, 'C_small': 12, '5_loose': 13, 'O': 16, 'C_large_loose': 17, 'H_loose': 18, 'D': 19})
    wrist_list = inv_dict({'D-CE': 0, 'D-EK': 1, 'D-KO': 2, 'D-O': 3, 'D-C': 4, '0': 5})
    extent_list = inv_dict({'0': 0, 'SMALL': 1, 'MEDIUM': 2, 'LARGE': 3})
    practice_list = inv_dict({'shaping': 1, 'indexing': 2, 'shaping-modelling': 3, 'grasping-indexing': 4, 'drawing': 5, 'modelling': 6, '0': 0, 'hedging': 7, 'grasping': 8, 'sizing': 9, 'counting': 10, 'action': 11, 'modelling-indexing': 12, 'shaping-sizing': 13})
    anno_id_list = [("Entity: ",entity_list),
                 ("Occurence: ",None),
                 ("Left Phase: ",phase_list),
                 ("Right Phase: ",phase_list),

                 ("Left Phrase: ", phrase_list),
                 ("Right Phrase: ", phrase_list),

                 ("Left Gesture Position: ", position_list),
                 ("Right Gesture Position: ", position_list),
                 ("Speech Gesture Position: ", position_list),

                 ("Left Hand Shape: ", shape_list),
                 ("Right Hand Shape: ", shape_list),

                 ("Left Wrist Distance: ", wrist_list),
                 ("Right Wrist Distance: ", wrist_list),

                 ("Left Extent: ", extent_list),
                 ("Right Extent: ", extent_list),

                 ("Left Practice: ", practice_list),
                 ("Right Practice: ", practice_list),
    ]
    print(anno.max())
    print(anno.min())

    for i in tqdm(range(output_poses.shape[0])):


        if withAnnotations:
            f1 = vis.VisUpperBody(output_poses[i].reshape((53, 3)), np.ones(53, ))
            f2 = vis.VisUpperBody(target_poses[i].reshape((53, 3)), np.ones(53, ))
            cv2.putText(f1, 'Generated:',(100, 100),cv2.FONT_HERSHEY_SIMPLEX,2,(255, 255, 255),2,2)
            cv2.putText(f2, 'Human:',(100, 100),cv2.FONT_HERSHEY_SIMPLEX,2,(255, 255, 255),2,2)
            img = np.concatenate((f2, f1), axis=1)
            img = np.pad(img,((0,960),(0,0),(0,0)))

            for h,an in enumerate(anno_id_list):
                aab = anno[i]
                va = int(aab[h])
                if an[1] is not None:
                    st_a = an[1][va] if va in an[1] and va > -1 else "None"
                    cv2.putText(img, an[0] + str(st_a), (100, 900+h*60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, 2)
                else:
                    cv2.putText(img, an[0] + str(format(float(va), '.2f')), (100, 900 + h * 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, 2)
        else:
            if save_gen:
                img = vis.VisUpperBody(output_poses[i].reshape((53, 3)), np.ones(53, ))
            else:
                img = vis.VisUpperBody(target_poses[i].reshape((53, 3)), np.ones(53, ))

        filename = "vis/{:06d}.jpg".format(i)
        cv2.imwrite(filename,img,[int(cv2.IMWRITE_JPEG_QUALITY), 98])
        comb_imgs.append(filename)

    # show audio
    audio_path = None
    if audio is not None:
        assert len(audio.shape) == 1  # 1-channel, raw signal
        audio = audio.astype(np.float32)
        sr = 16000
        audio_path = '{}/{}_{:03d}_{}.wav'.format(save_path, prefix, epoch, iter_idx)
        sf.write(audio_path, audio, sr)


    #save video
    try:
        video_path = 'temp_video.mp4'
        cmd = ['ffmpeg', '-y','-framerate', '15', '-i', 'vis/%06d.jpg', '-r','15','-crf','18', '-pix_fmt','yuv420p','temp_video.mp4']
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        print(result.returncode, result.stdout, result.stderr)
    except RuntimeError:
        traceback.print_exc()

    # merge audio and video
    if audio is not None:
        merged_video_path = 'test_full/{}.mp4'.format(video_filename)
        cmd = ['ffmpeg', '-y', '-i', audio_path,'-i', video_path,'-map','1:0','-map','0:0', '-strict', '-2',
               merged_video_path]
        if clipping_to_shortest_stream:
            cmd.insert(len(cmd) - 1, '-shortest')
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        print(result.returncode, result.stdout, result.stderr)
        os.remove(audio_path)
        os.remove(video_path)
        for f in os.listdir("vis/"):
            if f.endswith(".jpg"):
                os.remove("vis/"+f)

    print('done, took {:.1f} seconds'.format(time.time() - start))
    return output_poses, target_poses


def save_checkpoint(state, filename):
    torch.save(state, filename)
    logging.info('Saved the checkpoint')


def get_speaker_model(net):
    try:
        if hasattr(net, 'module'):
            speaker_model = net.module.z_obj
        else:
            speaker_model = net.z_obj
    except AttributeError:
        speaker_model = None

    if not isinstance(speaker_model, vocab.Vocab):
        speaker_model = None

    return speaker_model


def load_checkpoint_and_model(checkpoint_path, _device='cuda'):
    # print('loading checkpoint {}'.format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location=_device)
    args = checkpoint['args']
    epoch = checkpoint['epoch']
    lang_model = checkpoint['lang_model']
    speaker_model = checkpoint['speaker_model']
    pose_dim = checkpoint['pose_dim']
    print('epoch {}'.format(epoch))
    print(args)
    args.checkpoint_path = checkpoint_path

    generator, discriminator = init_beat_model(args, lang_model, speaker_model, pose_dim,load=True)
    generator = generator.to(_device)
    #if discriminator is not None:
    #    discriminator = discriminator.to(_device)
    loss_fn = torch.nn.L1Loss()

    #generator.load_state_dict(checkpoint['gen_dict'])

    # set to eval mode
    generator.train(False)

    return args, generator, loss_fn, lang_model, speaker_model, pose_dim


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
