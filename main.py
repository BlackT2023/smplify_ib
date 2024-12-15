import sys
import os

import os.path as osp

import time
import yaml
import smplx
import torch
import pickle
import logging
import numpy as np

from dataset.InBedPressureDataset import InBedPressureDataset

from utils.others.utils import NoteGeneration, setup_seed
from utils.others.loss_record import updateLoss

from torch.utils.data import DataLoader

from config.cmd_train_parser import parser_train_config
from models.smplx_body_vq import TrainWrapper as model

from models.basic_structure import VQVAE as s2g_body
from SMPLify.smplifyIBC import SMPLifyIBC

def main(args, task, weight):

    setup_seed(42)

    logging_path = os.path.join(args.logging_path, NoteGeneration(args))
    checkpoints_path = os.path.join(args.checkpoints_path, NoteGeneration(args))

    os.makedirs(logging_path, exist_ok=True)
    os.makedirs(checkpoints_path, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(filename)s %(levelname)s %(message)s',
                        datefmt='%a %d %b %Y %H:%M:%S', filename=os.path.join(logging_path, 'logging.log'),
                        filemode='w')
    logger = logging.getLogger(__name__)

    logging.info(f"Start PIMesh training for {args.epochs} epochs.")

    loss_record = updateLoss(logging_path)
    loss_record.start()

    cfgs = {
        'dataset_path': args.dataset_path,
        'save_dataset_path': '',
        'dataset_mode': args.exp_mode,
        'curr_fold': args.curr_fold,
        'seqlen': args.seqlen,
        'overlap': 0,
        'normalize': True,
        'img_size': args.pi_img_size,
        'repeat_num': 1
    }

    # get device and set dtype
    dtype = torch.float32
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # create Dataset from db file if available else create from folders
    data_set = InBedPressureDataset(
        cfgs,
        mode=task
    )

    loader = DataLoader(
        dataset=data_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2
    )

    # vae = s2g_body(69, embedding_dim=64, num_embeddings=2048, num_hiddens=1024,
    #                        num_residual_layers=2, num_residual_hiddens=512).to(device)
    vae = model(args).to(device)

    # vae_path = r'/workspace/wzy1999/projects/MotionOpt/data/pretrain/hps_102_losses_56.08_18.46.pth'
    # vae_path = r'/workspace/wzy1999/projects/MotionOpt/data/pretrain/hps_196_losses_57.61_28.58.pth'
    # vae_path = r'/workspace/wzy1999/projects/MotionOpt/data/pretrain/hps_193_losses_53.91_16.67.pth'
    # vae_path = r'/workspace/wzy1999/projects/MotionOpt/data/pretrain/hps_190_losses_53.04_17.49.pth'

    # vae_path = r'/workspace/wzy1999/projects/MotionOpt/data/pretrain/hps_76_losses_66.51_14.95.pth'

    # vae_path = r'/workspace/wzy1999/projects/MotionOpt/data/pretrain/transformer/hps_84_losses_26.61_10.06.pth'
    # vae_path = r'/workspace/wzy1999/projects/MotionOpt/data/pretrain/transformer/turn/hps_179_losses_31.99_22.23.pth'

    vae_path = r'/workspace/wzy1999/projects/MotionOpt/data/pretrain/transformer/64/hps_200_losses_32.69_9.33.pth'
    # vae_path = r'/workspace/wzy1999/projects/MotionOpt/data/pretrain/transformer/64_turn/hps_244_losses_34.23_14.78.pth'
    ckpt = torch.load(vae_path, map_location="cpu")['state_dict']
    vae.load_state_dict(ckpt)

    smplify = SMPLifyIBC(
        args=args,
        loader=loader,
        model=vae,
        step_size=args.lr,
        batch_size=args.seqlen,
        num_iters=args.num_smplify_iters,
        sensor_pitch=args.sensor_pitch,
        pressure_reso=args.sensor_size,
        weight=weight,
    )

    loss = smplify.fit()
    loss_record.end()

    return loss

if __name__ == '__main__':

    args = parser_train_config()
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(1)
    # args.gpu = 1
    # gpu = args.gpu
    args.note = '300_local_cosine'
    args.num_smplify_iters = 30
    # for i in [1e-1, 5e-1]:
    #     for j in [0.1, 1, 10, 100, 1000, 10000]:
    #         for k in [0.1, 1, 10, 100, 1000, 10000]:
    #             for l in [0.1, 1, 10, 100, 1000, 10000]:
    #                 args.lr = i
    #                 weight = {
    #                     'balance_loss': 0.5,
    #                     'smpl_loss_weight': j,
    #                     'joint_loss_weight': k,
    #                     'smooth_weight': l,
    #                     'pre_smooth_weight': l,
    #                 }
    #                 if i == 1e-1and j < 10:
    #                     continue
    #                 with open('result.txt', 'a') as w:
    #                     loss = main(args, 'test', weight)
    #
    #                     print(weight, loss)
    #                     w.write(f'{args.lr}, {weight}:   {loss}\n')

    weight = {
        'balance_loss': 0.5,
        'smpl_loss_weight': 0.1,
        'joint_loss_weight': 1,
        'smooth_weight': 0.1,
        'pre_smooth_weight': 0.1,
    }
    args.lr = 0.01
    with open('result.txt', 'a') as w:
        loss = main(args, 'test', weight)

        print(weight, loss)
        w.write(f'{args.lr}, {weight}:   {loss}\n')



