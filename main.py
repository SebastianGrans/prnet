#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import gc
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from data import ModelNet40
from data import Bunny
import numpy as np
from torch.utils.data import DataLoader
from model import PRNet



def _init_(args):
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)
    if not os.path.exists('checkpoints/' + args.exp_name + '/' + 'models'):
        os.makedirs('checkpoints/' + args.exp_name + '/' + 'models')
    os.system('cp main.py checkpoints' + '/' + args.exp_name + '/' + 'main.py.backup')
    os.system('cp model.py checkpoints' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp data.py checkpoints' + '/' + args.exp_name + '/' + 'data.py.backup')


def train(args, net, train_loader, test_loader):



    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(net.parameters(), lr=args.lr * 100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)



    if args.model_path is not '':
        assert os.path.exists(args.model_path), "Trying to resume, but model given doesn't exists."

        state = torch.load(args.model_path)
        net.set_state(state['model_state_dict'])
        opt.load_state_dict(state['optimizer_state_dict'])
        start_epoch = state['epoch'] + 1
        info_test_best = state['info_test_best']
        print("Resuming from previous state: %s" % args.model_path)
        print("Previous best: ")
        net.logger.write(info_test_best, write=False)
    else:
        start_epoch = 0
        info_test_best = None



    epoch_factor = args.epochs / 100.0

    scheduler = MultiStepLR(opt,
                            milestones=[int(30*epoch_factor), int(60*epoch_factor), int(80*epoch_factor)],
                            gamma=0.1)

    

    for epoch in range(start_epoch, args.epochs):
        info_train = net._train_one_epoch(epoch=epoch, train_loader=train_loader, opt=opt)
        scheduler.step()
        with torch.no_grad():
            info_test = net._test_one_epoch(epoch=epoch, test_loader=test_loader)


            savedict = {
                'epoch': epoch,
                'model_state_dict': net.get_state(),
                'optimizer_state_dict': opt.state_dict(),
                'info_test_best': info_test_best
            }


            if info_test_best is None or info_test_best['loss'] > info_test['loss']:
                info_test_best = info_test
                info_test_best['stage'] = 'best_test'
                savedict['info_test_best'] = info_test_best

                torch.save(savedict, 'checkpoints/%s/models/model.best.t7' % args.exp_name)

                # net.save('checkpoints/%s/models/model.best.t7' % args.exp_name)
            net.logger.write(info_test_best)

            torch.save(savedict, 'checkpoints/%s/models/model.%d.t7' % (args.exp_name, epoch))

            

        gc.collect()


def main():
    parser = argparse.ArgumentParser(description='Point Cloud Registration')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='prnet', metavar='N',
                        choices=['prnet'],
                        help='Model to use, [prnet]')
    parser.add_argument('--emb_nn', type=str, default='dgcnn', metavar='N',
                        choices=['pointnet', 'dgcnn'],
                        help='Embedding to use, [pointnet, dgcnn]')
    parser.add_argument('--attention', type=str, default='transformer',
                        metavar='N',
                        choices=['identity', 'transformer'],
                        help='Head to use, [identity, transformer]')
    parser.add_argument('--head', type=str, default='svd', metavar='N',
                        choices=['mlp', 'svd'],
                        help='Head to use, [mlp, svd]')
    parser.add_argument('--svd_on_gpu', action='store_true', default=False,
                        help='Run SVD on the GPU')
    parser.add_argument('--n_emb_dims', type=int, default=512, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--n_blocks', type=int, default=1, metavar='N',
                        help='Num of blocks of encoder&decoder')
    parser.add_argument('--n_heads', type=int, default=4, metavar='N',
                        help='Num of heads in multiheadedattention')
    parser.add_argument('--n_iters', type=int, default=3, metavar='N',
                        help='Num of iters to run inference')
    parser.add_argument('--discount_factor', type=float, default=0.9,
                        metavar='N', help='Discount factor to compute the loss')
    parser.add_argument('--n_ff_dims', type=int, default=1024, metavar='N',
                        help='Num of dimensions of fc in transformer')
    parser.add_argument('--n_keypoints', type=int, default=512, metavar='N',
                        help='Num of keypoints to use')
    parser.add_argument('--temp_factor', type=float, default=100, metavar='N',
                        help='Factor to control the softmax precision')
    parser.add_argument('--cat_sampler', type=str, default='gumbel_softmax',
                        choices=['softmax', 'gumbel_softmax'],
                        metavar='N', 
                        help='use gumbel_softmax to get the categorical sample')
    parser.add_argument('--dropout', type=float, default=0.0, metavar='N',
                        help='Dropout ratio in transformer')
    parser.add_argument('--batch_size', type=int, default=6,
                        metavar='batch_size', help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=12, 
                        metavar='batch_size', help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=False,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    
    # This isn't implemented.
    # parser.add_argument('--no_cuda', action='store_true', default=False,
    #                    help='enables CUDA training')

    # parser.add_argument('--seed', type=int, default=1234, metavar='S',
    #                    help='random seed')
    parser.add_argument('--seed', type=int, metavar='S',
                        help='random seed')

    parser.add_argument('--eval', action='store_true', default=False,
                        help='evaluate the model')
    parser.add_argument('--visualize', action='store_true', default=False,
                        help='Visualize the output of the network')

    parser.add_argument('--cycle_consistency_loss', type=float, default=0.1,
                        metavar='N', help='cycle consistency loss')
    parser.add_argument('--feature_alignment_loss', type=float, default=0.1,
                        metavar='N', help='feature alignment loss')
    parser.add_argument('--gaussian_noise', type=bool, default=False,
                        metavar='N', help='Wheter to add gaussian noise')
    parser.add_argument('--unseen', type=bool, default=False, metavar='N',
                        help='Wheter to test on unseen category')
    parser.add_argument('--n_points', type=int, default=1024, metavar='N',
                        help='Num of points to use')
    parser.add_argument('--n_subsampled_points', type=int, default=768,
                        metavar='N', help='Num of subsampled points to use')
    parser.add_argument('--dataset', type=str, default='modelnet40',
                        choices=['modelnet40', 'bunny'], metavar='N',
                        help='dataset to use')
    parser.add_argument('--rot_factor', type=float, default=4, metavar='N',
                        help='Divided factor of rotation')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path. Can be used to resume training, or to evaluate a specific checkpoint.')


    args = parser.parse_args()
    # torch.backends.cudnn.deterministic = True # Original
    # torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)

    _init_(args)


    if args.model == 'prnet':
        net = PRNet(args).cuda()
    else:
        raise Exception('Not implemented')

    if not args.eval:
        if args.dataset == 'modelnet40':
            train_loader = DataLoader(ModelNet40(num_points=args.n_points,
                                                num_subsampled_points=args.n_subsampled_points,
                                                partition='train', gaussian_noise=args.gaussian_noise,
                                                unseen=args.unseen, rot_factor=args.rot_factor),
                                    batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=6)
            test_loader = DataLoader(ModelNet40(num_points=args.n_points,
                                                num_subsampled_points=args.n_subsampled_points,
                                                partition='test', gaussian_noise=args.gaussian_noise,
                                                unseen=args.unseen, rot_factor=args.rot_factor),
                                    batch_size=args.test_batch_size, shuffle=False, drop_last=False, num_workers=6)
        else:
            raise Exception("Dataset not implemented")

        train(args, net, train_loader, test_loader)

    # Testing 
    # I've only implemented ModelNet40 so far.
    # This will simply load a single sample from the dataset 
    # and run net.predict()
    if args.eval:
        net.eval()
        with torch.no_grad():
            # If no model is specified, 
            if args.model_path is '':
                model_path = 'checkpoints' + '/' + args.exp_name + '/models/model.best.t7'
            else:
                model_path = args.model_path
            if not os.path.exists(model_path):
                print("Can't find pretrained model. Did you specify one with '--model_path'?")
                return

            state = torch.load(model_path)
            net.set_state(state['model_state_dict'])

            if args.dataset == 'modelnet40':
                dl = DataLoader(ModelNet40(num_points=args.n_points,
                                                num_subsampled_points=args.n_subsampled_points,
                                                partition='test', gaussian_noise=args.gaussian_noise,
                                                unseen=args.unseen, rot_factor=args.rot_factor),
                                    batch_size=1, shuffle=False, drop_last=False, num_workers=1)
            elif args.dataset == 'bunny':
                dl = DataLoader(
                        Bunny(num_subsampled_points=args.n_subsampled_points,
                            rot_factor=args.rot_factor
                        ),
                        batch_size=1,
                        shuffle=False,
                        drop_last=False,
                        num_workers=1)


            if args.visualize:
                # If you try to import Open3D while running "Remote 
                # Development" with VS Code I get an error. That's why I'm 
                # importing it in the middle of the code like this... 
                import open3d as o3d


            for data in dl:
                src, tgt, rotation_ab, translation_ab, rotation_ba, translation_ba, euler_ab, euler_ba = [d.cuda() for d in data]

                # Here we apply the network to the src and tgt point cloud. 
                # By default, this will run three iterations. 
                rotation_ab_pred, translation_ab_pred = net.predict(src, tgt, n_iters=6)
                
                print("True transformation:")
                print(rotation_ab)
                print(translation_ab)
                print("------------------------")
                print("Predicted transformation:")
                print(rotation_ab_pred)
                print(translation_ab_pred)
                print("------------------------")

                # Here I use Open3D to visualize the point clouds before and 
                # after transformation with the predicted transformation.
                if args.visualize:
                    R = rotation_ab_pred[0].cpu().numpy()
                    Tr = translation_ab_pred.cpu().numpy()
                    T = np.hstack((R, Tr.T))
                    T = np.vstack((T, [0, 0, 0, 1]))

                    src_o3d = src[0].cpu().numpy().T
                    tgt_o3d = tgt[0].cpu().numpy().T
                    srcv = o3d.utility.Vector3dVector(src_o3d)
                    tgtv = o3d.utility.Vector3dVector(tgt_o3d)
                    srcpcd = o3d.geometry.PointCloud(srcv)
                    tgtpcd = o3d.geometry.PointCloud(tgtv)

                    srcpcd.paint_uniform_color([0, 1, 0])
                    tgtpcd.paint_uniform_color([1, 0, 0])
                    o3d.visualization.draw_geometries([srcpcd, tgtpcd])

                    srcpcd = srcpcd.transform(T)
                    o3d.visualization.draw_geometries([srcpcd, tgtpcd])

                
    print('FINISH')


if __name__ == '__main__':
    main()
