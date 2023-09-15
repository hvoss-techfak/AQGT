import configargparse


def str2bool(v):
    """ from https://stackoverflow.com/a/43357954/1361529 """
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise configargparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = configargparse.ArgParser()
    parser.add('-c', '--config', required=True, is_config_file=True, help='Config file path')
    parser.add("--name", type=str, default="main", help='Name of the project. Currently unused.')
    parser.add("--train_data_path", action="append", help='Path to the train data lmdb')
    parser.add("--val_data_path", action="append", help='Path to the validation data lmdb')
    parser.add("--test_data_path", action="append", help='Path to the test data lmdb')
    parser.add("--beat_data_path", action="append", help='Temporary path to store the generated beat data for the created SEEG data')
    parser.add("--web_data_path", action="append", help='Path for the created web data folder')
    parser.add("--model_save_path", required=True, help='Model output folder')
    parser.add("--pose_representation", type=str, default='3d_vec')
    parser.add("--mean_dir_vec", action="append", type=float, nargs='*')
    parser.add("--mean_pose", action="append", type=float, nargs='*')
    parser.add("--random_seed", type=int, default=-1)
    parser.add("--save_result_video", type=str2bool, default=True)
    parser.add("--checkpoint_path",type=str,default="None", help="Path of a checkpoint that should be loaded")
    parser.add("--max_v",type=int,default=2411)

    # word embedding
    parser.add("--wordembed_path", type=str, default=None)
    parser.add("--wordembed_dim", type=int, default=100)
    parser.add("--freeze_wordembed", type=str2bool, default=False)
    parser.add("--wandb_key",type=str,required=True)

    # model
    parser.add("--model", type=str, required=True)
    parser.add("--epochs", type=int, default=10)
    parser.add("--batch_size", type=int, default=50)
    parser.add("--dropout_prob", type=float, default=0.3)
    parser.add("--n_layers", type=int, default=2)
    parser.add("--hidden_size", type=int, default=200)
    parser.add("--z_type", type=str, default='none')
    parser.add("--input_context", type=str, default='both')

    # dataset
    parser.add("--motion_resampling_framerate", type=int, default=24)
    parser.add("--n_poses", type=int, default=50)
    parser.add("--n_pre_poses", type=int, default=5)
    parser.add("--subdivision_stride", type=int, default=5)
    parser.add("--loader_workers", type=int, default=16)

    # GAN parameter
    parser.add("--GAN_noise_size", type=int, default=0)

    # training
    parser.add("--learning_rate", type=float, default=0.001)
    parser.add("--discriminator_lr_weight", type=float, default=0.2)
    parser.add("--loss_regression_weight", type=float, default=50)
    parser.add("--loss_gan_weight", type=float, default=1.0)
    parser.add("--loss_kld_weight", type=float, default=0.1)
    parser.add("--loss_reg_weight", type=float, default=0.01)
    parser.add("--loss_vel_weight", type=float, default=0.1)

    parser.add("--loss_warmup", type=int, default=-1)

    # eval
    parser.add("--eval_net_path", type=str, default='')

    args = parser.parse_args()
    return args
