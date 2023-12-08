import argparse
import os
import datetime
import torch

class Options():
    ''' This class defines argsions
        adapted from: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/tree/master/options
    '''
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        ''' set up arguments '''

        parser.add_argument('--model_name', type = str, default='FCRN_ResNet50')
        parser.add_argument('--dataset_name', type = str, default='NYUv2', help='dataset')
        parser.add_argument('--loss_fnc', type = str, default='MSE')
        parser.add_argument('--decoder', type = str, default='uproj')
        parser.add_argument('--load_model_dir', type = str, default='')


        parser.add_argument('--bs', type = int, default=16)
        parser.add_argument('--lr', type = float, default=1e-2)
        parser.add_argument('--wd', type = float, default=5e-4)
        parser.add_argument('--mom', type = float, default=0.9)

        parser.add_argument('--device', type = str, default='cuda:0')
        parser.add_argument('--cpu_seed', type = int, default=24)
        parser.add_argument('--gpu_seed', type = int, default=32)

        self.initialized = True
        return parser

    def get_argsions(self):
        ''' get argsions from parser '''
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(description='Asyncrhonous toy example')
            parser = self.initialize(parser)

        # save and return the parser
        self.parser = parser
        return parser.parse_args()
    
    def parse(self):
        ''' Parse our argsions, create checkpoints directory suffix, and set up gpu device. '''
        args = self.get_argsions()
 
        if args.dataset_name.lower() == 'sun-rgbd':
            args.dataset = 'SUN-RGBD'
        elif args.dataset_name.lower() == 'nyuv2':
            args.dataset = 'NYUv2'
        elif args.dataset_name.lower() == 'diode':
            args.dataset = 'DIODE'
        else:
            raise ValueError("wrong dataset name!")
        
        self.args = args
        return self.args