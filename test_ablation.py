#!/usr/bin/env python
from __future__ import print_function

import argparse
import inspect
import os
import pickle
import random
import shutil
import sys
import time
from collections import OrderedDict
import traceback
from sklearn.metrics import confusion_matrix
import csv
import numpy as np
import glob
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve, auc, roc_curve
# torch
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from tensorboardX import SummaryWriter
from tqdm import tqdm

from torchlights.torchlight import DictAction


import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))
class MMD_loss(nn.Module):
    def __init__(self, kernel_mul = 2.0, kernel_num = 5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        return
    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2) 
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY -YX)
        return loss
def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' % (class_str, traceback.format_exception(*sys.exc_info())))

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(
        description='Spatial Temporal Graph Convolution Network')
    parser.add_argument(
        '--work-dir',
        default='./work_dir/temp',
        help='the work folder for storing results')

    parser.add_argument('-model_saved_name', default='')
    parser.add_argument(
        '--config',
        default='./config/nturgbd-cross-view/test_bone.yaml',
        help='path to the configuration file')

    # processor
    parser.add_argument(
        '--phase', default='test', help='must be train or test')
    parser.add_argument(
        '--save-score',
        type=str2bool,
        default=False,
        help='if ture, the classification score will be stored')

    # visulize and debug
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed for pytorch')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=100,
        help='the interval for printing messages (#iteration)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=1,
        help='the interval for storing models (#iteration)')
    parser.add_argument(
        '--save-epoch',
        type=int,
        default=30,
        help='the start epoch to save model (#iteration)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=1,
        help='the interval for evaluating models (#iteration)')
    parser.add_argument(
        '--print-log',
        type=str2bool,
        default=True,
        help='print logging or not')
    parser.add_argument(
        '--show-topk',
        type=int,
        default=[1, 5],
        nargs='+',
        help='which Top K accuracy will be shown')

    # feeder
    parser.add_argument(
        '--feeder', default='feeder.feeder', help='data loader will be used')
    parser.add_argument(
        '--num-worker',
        type=int,
        default=32,
        help='the number of worker for data loader')
    parser.add_argument(
        '--train-feeder-args',
        action=DictAction,
        default=dict(),
        help='the arguments of data loader for training')
    parser.add_argument(
        '--test-feeder-args-unseen',
        action=DictAction,
        default=dict(),
        help='the arguments of data loader for test')
    parser.add_argument(
        '--test-feeder-args-seen',
        action=DictAction,
        default=dict(),
        help='the arguments of data loader for test')
    # model
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument(
        '--model-args',
        action=DictAction,
        default=dict(),
        help='the arguments of model')
    parser.add_argument(
        '--model-args-upper',
        action=DictAction,
        default=dict(),
        help='the arguments of model')
    parser.add_argument(
        '--model-args-lower',
        action=DictAction,
        default=dict(),
        help='the arguments of model')
    parser.add_argument(
        '--weights',
        default='/pfs/work8/workspace/ffuc/scratch/fy2374-acmmm/aaai/checkpoints/CTRGCN_softmax_split1_bodypart/runs-50-10400.pt',
        help='the weights for network initialization')
    parser.add_argument(
        '--weights_upper',
        default='/pfs/work8/workspace/ffuc/scratch/fy2374-acmmm/aaai/checkpoints/CTRGCN_softmax_split1_bodypart/runs-upper50-10400.pt',
        help='the weights for network initialization')
    parser.add_argument(
        '--weights_lower',
        default='/pfs/work8/workspace/ffuc/scratch/fy2374-acmmm/aaai/checkpoints/CTRGCN_softmax_split1_bodypart/runs-lowwer50-10400.pt',
        help='the weights for network initialization')
    parser.add_argument(
        '--ignore-weights',
        type=str,
        default=[],
        nargs='+',
        help='the name of weights which will be ignored in the initialization')

    # optim
    parser.add_argument(
        '--base-lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument(
        '--step',
        type=int,
        default=[20, 40, 60],
        nargs='+',
        help='the epoch where optimizer reduce the learning rate')
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        nargs='+',
        help='the indexes of GPUs for training or testing')
    parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
    parser.add_argument(
        '--nesterov', type=str2bool, default=False, help='use nesterov or not')
    parser.add_argument(
        '--batch-size', type=int, default=256, help='training batch size')
    parser.add_argument(
        '--test-batch-size', type=int, default=256, help='test batch size')
    parser.add_argument(
        '--start-epoch',
        type=int,
        default=0,
        help='start training from which epoch')
    parser.add_argument(
        '--num-epoch',
        type=int,
        default=80,
        help='stop training in which epoch')
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.0005,
        help='weight decay for optimizer')
    parser.add_argument(
        '--lr-decay-rate',
        type=float,
        default=0.1,
        help='decay rate for learning rate')
    parser.add_argument('--warm_up_epoch', type=int, default=0)

    return parser


class Processor():
    """ 
        Processor for Skeleton-based Action Recgnition
    """

    def __init__(self, arg):
        self.arg = arg
        self.save_arg()
        if arg.phase == 'train':
            if not arg.train_feeder_args['debug']:
                arg.model_saved_name = os.path.join(arg.work_dir, 'runs')
                if os.path.isdir(arg.model_saved_name):
                    print('log_dir: ', arg.model_saved_name, 'already exist')
                    answer = input('delete it? y/n:')
                    if answer == 'y':
                        shutil.rmtree(arg.model_saved_name)
                        print('Dir removed: ', arg.model_saved_name)
                        input('Refresh the website of tensorboard by pressing any keys')
                    else:
                        print('Dir not removed: ', arg.model_saved_name)
                self.train_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'train'), 'train')
                self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'val'), 'val')
            else:
                self.train_writer = self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'test'), 'test')
        self.global_step = 0
        # pdb.set_trace()
        self.load_model()

        if self.arg.phase == 'model_size':
            pass
        else:
            self.load_optimizer()
            self.load_data()
        self.lr = self.arg.base_lr
        self.best_acc = 0
        self.best_acc_epoch = 0
        self.model = self.model.cuda(self.output_device)
        self.mmd_loss = MMD_loss().cuda()
        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                self.model = nn.DataParallel(
                    self.model,
                    device_ids=self.arg.device,
                    output_device=self.output_device)

    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()
        if self.arg.phase == 'train':
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker,
                drop_last=True,
                worker_init_fn=init_seed)
        self.data_loader['test_seen'] = torch.utils.data.DataLoader(
            dataset=Feeder(**self.arg.test_feeder_args_seen),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            worker_init_fn=init_seed)
        self.data_loader['test_unseen'] = torch.utils.data.DataLoader(
            dataset=Feeder(**self.arg.test_feeder_args_unseen),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            worker_init_fn=init_seed)
    def load_model(self):
        output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        self.output_device = output_device
        Model = import_class(self.arg.model)
        shutil.copy2(inspect.getfile(Model), self.arg.work_dir)
        print(Model)
        self.model = Model(**self.arg.model_args)
        self.model_upperbody = Model(**self.arg.model_args)
        self.model_lowerbody = Model(**self.arg.model_args)
        print(self.model)
        self.loss = nn.CrossEntropyLoss().cuda(output_device)

        self.model_upperbody = self.model_upperbody.cuda()
        self.model_lowerbody = self.model_lowerbody.cuda()
        if self.arg.weights:
            #self.global_step = int(arg.weights[:-3].split('-')[-1])
            self.print_log('Load weights from {}.'.format(self.arg.weights))
            if '.pkl' in self.arg.weights:
                with open(self.arg.weights, 'r') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg.weights)

            weights = OrderedDict([[k.split('module.')[-1], v.cuda(output_device)] for k, v in weights.items()])

            keys = list(weights.keys())
            for w in self.arg.ignore_weights:
                for key in keys:
                    if w in key:
                        if weights.pop(key, None) is not None:
                            self.print_log('Sucessfully Remove Weights: {}.'.format(key))
                        else:
                            self.print_log('Can Not Remove Weights: {}.'.format(key))
            
            self.model.load_state_dict(weights)
            if '.pkl' in self.arg.weights_upper:
                with open(self.arg.weights_upper, 'r') as f:
                    weights_upper = pickle.load(f)
            else:
                weights_upper = torch.load(self.arg.weights_upper)

            weights_upper = OrderedDict([[k.split('module.')[-1], v.cuda(output_device)] for k, v in weights_upper.items()])

            keys = list(weights_upper.keys())
            for w in self.arg.ignore_weights:
                for key in keys:
                    if w in key:
                        if weights_upper.pop(key, None) is not None:
                            self.print_log('Sucessfully Remove Weights: {}.'.format(key))
                        else:
                            self.print_log('Can Not Remove Weights: {}.'.format(key))
            self.model_upperbody.load_state_dict(weights_upper)
            if '.pkl' in self.arg.weights_lower:
                with open(self.arg.weights_lower, 'r') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg.weights_lower)

            weights = OrderedDict([[k.split('module.')[-1], v.cuda(output_device)] for k, v in weights.items()])

            keys = list(weights.keys())
            for w in self.arg.ignore_weights:
                for key in keys:
                    if w in key:
                        if weights.pop(key, None) is not None:
                            self.print_log('Sucessfully Remove Weights: {}.'.format(key))
                        else:
                            self.print_log('Can Not Remove Weights: {}.'.format(key))

            self.model_lowerbody.load_state_dict(weights)

        self.model = torch.nn.DataParallel(self.model)
        self.model_upperbody = torch.nn.DataParallel(self.model_upperbody)
        self.model_lowerbody = torch.nn.DataParallel(self.model_lowerbody)
    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                [*self.model.parameters()] + [*self.model_lowerbody.parameters()] + [*self.model_upperbody.parameters()],
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                [*self.model.parameters()] + [*self.model_lowerbody.parameters()] + [*self.model_upperbody.parameters()],
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

        self.print_log('using warm up, epoch: {}'.format(self.arg.warm_up_epoch))

    def save_arg(self):
        # save arg
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            f.write(f"# command line: {' '.join(sys.argv)}\n\n")
            yaml.dump(arg_dict, f)

    def adjust_learning_rate(self, epoch):
        if self.arg.optimizer == 'SGD' or self.arg.optimizer == 'Adam':
            if epoch < self.arg.warm_up_epoch:
                lr = self.arg.base_lr * (epoch + 1) / self.arg.warm_up_epoch
            else:
                lr = self.arg.base_lr * (
                        self.arg.lr_decay_rate ** np.sum(epoch >= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return lr
        else:
            raise ValueError()

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(str, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def train(self, epoch, save_model=False):
        self.model.train()
        self.print_log('Training epoch: {}'.format(epoch + 1))
        loader = self.data_loader['train']
        self.adjust_learning_rate(epoch)
        loss_value = []
        acc_value = []
        self.train_writer.add_scalar('epoch', epoch, self.global_step)
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)
        process = tqdm(loader, ncols=40)

        for batch_idx, (datalist, label, index) in enumerate(process):
            self.global_step += 1
            with torch.no_grad():
                data = datalist[0].float().cuda(self.output_device)
                data_lower = datalist[1].float().cuda(self.output_device)
                data_upper = datalist[2].float().cuda(self.output_device)
                label = label.long().cuda(self.output_device)
            timer['dataloader'] += self.split_time()
            # forward
            output = self.model(data)
            #print(data_upper.shape)
            output_upper = self.model_upperbody(data_upper)
            output_lower = self.model_lowerbody(data_lower)
            loss = self.loss(output, label) + self.loss(output_lower, label) + torch.mean(self.loss(output_upper, label)) + torch.mean(self.mmd_loss(output, output_lower)) + torch.mean(self.mmd_loss(output, output_upper)) + torch.mean(self.mmd_loss(output_upper, output_lower))
            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_value.append(loss.data.item())
            timer['model'] += self.split_time()

            value, predict_label = torch.max(output.data, 1)
            acc = torch.mean((predict_label == label.data).float())
            acc_value.append(acc.data.item())
            self.train_writer.add_scalar('acc', acc, self.global_step)
            self.train_writer.add_scalar('loss', loss.data.item(), self.global_step)

            # statistics
            self.lr = self.optimizer.param_groups[0]['lr']
            self.train_writer.add_scalar('lr', self.lr, self.global_step)
            timer['statistics'] += self.split_time()
            #break
        # statistics of time consumption and loss
        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }
        self.print_log(
            '\tMean training loss: {:.4f}.  Mean training acc: {:.2f}%.'.format(np.mean(loss_value), np.mean(acc_value)*100))
        self.print_log('\tTime consumption: [Data]{dataloader}, [Network]{model}'.format(**proportion))

        if save_model:
            state_dict = self.model.state_dict()
            weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])
            state_dict_upper = self.model_upperbody.state_dict()
            weights_upper = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict_upper.items()])
            state_dict_lower = self.model_lowerbody.state_dict()
            weights_lower = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict_lower.items()])
            torch.save(weights, self.arg.model_saved_name + '-' + str(epoch+1) + '-' + str(int(self.global_step)) + '.pt')
            torch.save(weights_upper, self.arg.model_saved_name + '-upper' + str(epoch+1) + '-' + str(int(self.global_step)) + '.pt')
            torch.save(weights_lower, self.arg.model_saved_name + '-lowwer' + str(epoch+1) + '-' + str(int(self.global_step)) + '.pt')
    def eval_osr(self, y_true, y_pred):
        # open-set auc-roc (binary class)
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
        #print(y_true.shape)
        #print(y_pred.shape)
        auroc = roc_auc_score(y_true, y_pred)

        # open-set auc-pr (binary class)
        # as an alternative, you may also use `ap = average_precision_score(labels, uncertains)`, which is approximate to aupr.
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        aupr = auc(recall, precision)

        # open-set fpr@95 (binary class)
        fpr, tpr, _ = roc_curve(y_true, y_pred, pos_label=1)
        operation_idx = np.abs(tpr - 0.95).argmin()
        fpr95 = fpr[operation_idx]  # FPR when TPR at 95%
        return auroc, aupr, fpr95
    def eval_uosr(self, y_true, y_pred):
        # open-set auc-roc (binary class)
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
        auroc = roc_auc_score(y_true, y_pred)
        # open-set auc-pr (binary class)
        # as an alternative, you may also use `ap = average_precision_score(labels, uncertains)`, which is approximate to aupr.
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        aupr = auc(recall, precision)
        # open-set fpr@95 (binary class)
        fpr, tpr, _ = roc_curve(y_true, y_pred, pos_label=1)
        operation_idx = np.abs(tpr - 0.95).argmin()
        fpr95 = fpr[operation_idx]  # FPR when TPR at 95%
        return auroc, aupr, fpr95
    def eval(self, epoch, save_score=False, loader_name=['test_seen', 'test_unseen'], wrong_file=None, result_file=None, num_class=None):
        num_class = 40
        threshold_v = 0.9
        threshold_m = 0.9
        total = torch.zeros(num_class+1)
        correct_mean_seen = torch.zeros(num_class+1)
        correct_var_seen = torch.zeros(num_class+1)
        correct_mean_unseen = torch.zeros(num_class+1)
        correct_var_unseen = torch.zeros(num_class+1)
        all_prob_seen = []
        all_preds_seen = []
        all_labels_seen = []
        all_prob_unseen = []
        all_labels_unseen = []
        all_preds_unseen = []
        if wrong_file is not None:
            f_w = open(wrong_file, 'w')
        if result_file is not None:
            f_r = open(result_file, 'w')
        self.model.eval()
        self.print_log('Eval epoch: {}'.format(epoch + 1))
        step = 0
        process = tqdm(self.data_loader['test_seen'], ncols=40)
        for batch_idx, (datalist, label,index) in enumerate(process):
            with torch.no_grad():
                #output = []
                data = datalist[0].float().cuda(self.output_device)
                data_lower = datalist[1].float().cuda(self.output_device)
                data_upper = datalist[2].float().cuda(self.output_device)
                label = label.long().cuda(self.output_device)
                output = self.model(data)
                output_lower = self.model_lowerbody(data_lower)
                output_upper = self.model_upperbody(data_upper)
                mean_indicator = (output + output_lower + output_upper)/3
                output = torch.nn.functional.softmax(output/2, dim = -1)
                #output_lower = torch.nn.functional.softmax(output_lower/2, dim = -1)
                #output_upper = torch.nn.functional.softmax(output_upper/2, dim = -1)
                
                probab, predicted = torch.max(output, -1)
                all_preds_seen.append(predicted)
                all_prob_seen.append(probab)
                all_labels_seen.append(label)
                #probab = torch.ones_like(cal_output).cuda()
                #print(probab)
                for k in range(len(predicted)):
                    total[label[k]] += 1
                    '''if (probab[k] < threshold_m):
                        predicted[k] = len(total) - 1'''
                    if predicted[k] == label[k]:
                        correct_mean_seen[predicted[k]] += 1
                    if predicted[k] == label[k]:
                        correct_var_seen[predicted[k]] += 1
        all_prob_seen = torch.cat(all_prob_seen, 0)
        all_labels_seen = torch.cat(all_labels_seen,0)
        all_preds_seen = torch.cat(all_preds_seen)
        process = tqdm(self.data_loader['test_unseen'], ncols=40)
        for batch_idx, (datalist, label,index) in enumerate(process):
            with torch.no_grad():
                #output = []
                data = datalist[0].float().cuda(self.output_device)
                data_lower = datalist[1].float().cuda(self.output_device)
                data_upper = datalist[2].float().cuda(self.output_device)
                label = label.long().cuda(self.output_device)
                output = self.model(data)
                output_lower = self.model_lowerbody(data_lower)
                output_upper = self.model_upperbody(data_upper)
                output = (output + output_lower + output_upper)/3
                output = torch.nn.functional.softmax(output/2, dim = -1)
                #output_lower = torch.nn.functional.softmax(output_lower/2, dim = -1)
                #output_upper = torch.nn.functional.softmax(output_upper/2, dim = -1)
                

                #probab = torch.ones_like(cal_output).cuda()#cal_output
                #p#rint(probab)
                #output_mean = torch.mean(output, dim = 0)
                #print(output.max(-1)[0])
                #print(output_lower.max(-1)[0])
                #print(output_upper.max(-1)[0])
                probab, predicted = torch.max(output, 1)
                all_preds_unseen.append(predicted)
                all_prob_unseen.append(probab)
                all_labels_unseen.append(label)

                for k in range(len(predicted)):
                    '''if (probab[k] < threshold_m):
                        predicted[k] = 10'''
                    if predicted[k] == label[k]:
                        correct_mean_unseen[predicted[k]] += 1

        all_prob_unseen = torch.cat(all_prob_unseen, 0)
        all_labels_unseen = torch.cat(all_labels_unseen, 0)
        all_preds_unseen = torch.cat(all_preds_unseen, 0)              
        N = all_prob_seen.shape[0] #+ all_labels_seen.shape[0]
        correct_mean = correct_mean_seen #+ correct_mean_unseen
        mixed_acc = torch.sum(correct_mean)/N * 100
        ###############################calculate OS auc############################
        all_prob = torch.cat([all_prob_seen, all_prob_unseen])
        all_prob = 1 - all_prob
        binary_label_uncertainty = torch.cat([torch.zeros(all_labels_seen.shape[0]), torch.ones(all_labels_unseen.shape[0])], 0)
        auroc, aupr, fpr95 = self.eval_osr(y_true=binary_label_uncertainty, y_pred=all_prob)
        ###############################calculate UOS auc###########################
        N = all_labels_seen.shape[0]
        topK = N - int(N*0.85)
        uncertainty_seen = 1- all_prob_seen
        threshold = torch.sort(uncertainty_seen, 0)[0][N-topK+1]
        #all_preds_seen[uncertainty_seen>threshold] = torch.ones(all_preds_seen[uncertainty_seen>threshold].shape[0]).cuda().long()
        #all_preds_seen[uncertainty_seen<threshold] = torch.zeros(all_preds_seen[uncertainty_seen<threshold].shape[0]).cuda().long()
        inc_labels = torch.zeros(all_preds_seen[uncertainty_seen<=threshold].shape[0])
        inw_labels = torch.ones(all_preds_seen[uncertainty_seen>threshold].shape[0])
        labels_seen = torch.cat([inc_labels, inw_labels], 0)
        preds_seen = torch.cat([uncertainty_seen[uncertainty_seen<=threshold], uncertainty_seen[uncertainty_seen>threshold]], 0)
        preds = torch.cat([preds_seen, 1-all_prob_unseen], 0)
        labels_uosr = torch.cat([labels_seen.cuda(),torch.ones(all_labels_unseen.shape[0]).cuda()], 0)
        auroc_uosr, aupr_uosr, fpr95_uosr = self.eval_uosr(y_true=labels_uosr, y_pred=preds)
        print('####Epoch: ', epoch+1, ' -----ACC mixed: ', mixed_acc, ' ------osauc:', auroc, ' ------osauc_uosr:', auroc_uosr)
        torch.save(self.model.state_dict(), self.arg.work_dir + '/checkpoints_epoch_{}.pt'.format(epoch))
        with open('{}/each_epoch_resuts.csv'.format(self.arg.work_dir), 'w') as f:
            writer = csv.writer(f)
            writer.writerow('Epoch_{}_MixedAcc_{}_OSAUC_{}_UOSAUC_{}'.format(epoch, mixed_acc, auroc, auroc_uosr))

    def start(self):
        if self.arg.phase == 'train':
            self.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            self.global_step = self.arg.start_epoch * len(self.data_loader['train']) / self.arg.batch_size
            def count_parameters(model):
                return sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.print_log(f'# Parameters: {count_parameters(self.model)}')
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                save_model = (((epoch + 1) % self.arg.save_interval == 0) or (
                        epoch + 1 == self.arg.num_epoch)) and (epoch+1) > self.arg.save_epoch

                self.train(epoch, save_model=save_model)
                if (epoch % self.arg.eval_interval == 0) and (epoch>5):
                    self.eval(epoch, save_score=self.arg.save_score, loader_name=['test_unseen', 'test_seen'])

            # test the best model
            weights_path = glob.glob(os.path.join(self.arg.work_dir, 'runs-'+str(self.best_acc_epoch)+'*'))[0]
            weights = torch.load(weights_path)
            if type(self.arg.device) is list:
                if len(self.arg.device) > 1:
                    weights = OrderedDict([['module.'+k, v.cuda(self.output_device)] for k, v in weights.items()])
            self.model.load_state_dict(weights)

            wf = weights_path.replace('.pt', '_wrong.txt')
            rf = weights_path.replace('.pt', '_right.txt')
            self.arg.print_log = False
            self.eval(epoch=0, save_score=True, loader_name=['test'], wrong_file=wf, result_file=rf)
            self.arg.print_log = True


            num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            self.print_log(f'Best accuracy: {self.best_acc}')
            self.print_log(f'Epoch number: {self.best_acc_epoch}')
            self.print_log(f'Model name: {self.arg.work_dir}')
            self.print_log(f'Model total number of params: {num_params}')
            self.print_log(f'Weight decay: {self.arg.weight_decay}')
            self.print_log(f'Base LR: {self.arg.base_lr}')
            self.print_log(f'Batch Size: {self.arg.batch_size}')
            self.print_log(f'Test Batch Size: {self.arg.test_batch_size}')
            self.print_log(f'seed: {self.arg.seed}')

        elif self.arg.phase == 'test':
            wf = self.arg.weights.replace('.pt', '_wrong.txt')
            rf = self.arg.weights.replace('.pt', '_right.txt')

            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.arg.print_log = False
            self.print_log('Model:   {}.'.format(self.arg.model))
            self.print_log('Weights: {}.'.format(self.arg.weights))
            self.eval(epoch=0, save_score=self.arg.save_score, loader_name=['test'], wrong_file=wf, result_file=rf)
            self.print_log('Done.\n')

if __name__ == '__main__':
    parser = get_parser()

    # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    init_seed(arg.seed)
    processor = Processor(arg)
    processor.start()
