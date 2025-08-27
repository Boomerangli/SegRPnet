import os
import time
import numpy as np

from tqdm import tqdm
import shutil
import random

import torch
from torch import nn
from torch.backends import cudnn
from torch.autograd import Variable

from agents.base import BaseAgent

# import your classes here

from tensorboardX import SummaryWriter
from utils.misc import print_cuda_statistics
from graphs.models.unet import Unet
from datasets.unet import UnetDataLoader
from torchvision.utils import save_image
from utils.metrix_psy import Metrix
from utils.help_funcs import get_thop_params_flops_Seg

cudnn.benchmark = True

class UnetAgent(BaseAgent):
    
    def __init__(self, config):
        super(UnetAgent, self).__init__(config)
        
        # 定义模型
        self.model = Unet()
        # 定义 data_loader
        self.data_loader = UnetDataLoader(config=config)
        # 定义 loss
        self.loss = nn.BCEWithLogitsLoss()
        # 定义optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate, betas=(self.config.beta1, self.config.beta2), weight_decay=self.config.weight_decay)
        
        # 初始化counter
        self.current_epoch = 0
        self.current_iteration = 0
        
        self.img_num = len(self.data_loader.train_loader)
        
        # 设置cuda
        self.is_cuda = torch.cuda.is_available()
        if self.is_cuda and not self.config.cuda:
            self.logger.info("WARNING: You have a CUDA device, so you should probably enable CUDA")

        self.cuda = self.is_cuda & self.config.cuda

        self.seed = self.config.seed
        self.device = torch.device('cuda')
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)  # Numpy module.
        random.seed(self.seed)  # Python random module.
        torch.manual_seed(self.seed)
        
        if self.cuda:
            self.model = self.model.to(self.device)
            self.loss = self.loss.to(self.device)
            
            self.logger.info("Program will run on *****GPU***** ")
            print_cuda_statistics()
        else:
            self.device = torch.device('cpu')
            
            self.logger.info("Program will run on *****CPU***** ")
            
        # 加载checkpoint
        # self.load_checkpoint(self.config.checkpoint_dir)
        # Summary_writer        
        self.summary_writer = SummaryWriter(log_dir=self.config.summary_dir, comment='Unet')
        
    def load_checkpoint(self, file_name):
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        :return:
        """
        
        

    def save_checkpoint(self):
        """
        Checkpoint saver
        :param file_name: name of the checkpoint file
        :param is_best: boolean flag to indicate whether current checkpoint's accuracy is the best so far
        :return:
        """     
        state = {'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(), 'epoch': self.current_epoch}
        model_name = self.config.model_name
        new_save = os.path.join(self.config.save_checkpoint_path, model_name + '_{}.pth'.format(self.config.max_epoch))
        torch.save(state, new_save)
        
         
    def run(self):
        try:
            self.train()
        
        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C.. wait to finalize")
    
    def train(self):
        for epoch in range(self.config.start_epoch, self.config.max_epoch):
            self.train_one_epoch()
            self.current_epoch += 1
            self.validate()
            
            if epoch % self.config.save_every == 0:
                self.save_checkpoint()
            
    def train_one_epoch(self):
        time1 = time.time()
        self.model.train()
        a = 0
        step = 0
        for batch_idx, (img, label) in enumerate(self.data_loader.train_loader):
            label = label.to(self.device)
            img = img.to(self.device)
            self.optimizer.zero_grad()
            out = self.model(img)

            loss = self.loss(out, label)
            loss.backward()
            self.optimizer.step()
            a += loss
            step += 1
            
            if batch_idx % self.config.log_interval == 0:
                self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    self.current_epoch, batch_idx * len(img), len(self.data_loader.train_loader.dataset),
                           100. * batch_idx / self.img_num, loss.item()))
            self.current_iteration += 1
            
        loss_avg = a / step
        if not os.path.exists(self.config.log_path):
            os.makedirs(self.config.log_path)

        with open(self.config.log_path + "log.txt", 'a+') as f:
            f.write("epoch :{} loss:{:.4f}\n".format((self.current_epoch + 1), loss_avg))
            f.close()

        time2 = time.time()
        print('this epoch take {}s'.format(time2 - time1))
    
    def validate(self):
        self.model.eval()
        with torch.no_grad():
            print(self.data_loader.test_loader)
            for i, (input_A, label) in enumerate(self.data_loader.val_loader):
                input_A = Variable(torch.FloatTensor(input_A))
                label = Variable(torch.FloatTensor(label))
                input_A, label = input_A.cuda(), label.cuda()
                out = self.model(input_A)
                label = torch.squeeze(label, 1)
                out = torch.sigmoid(out)
                # 输出二值化的值
                # out = tensor_binary(out)
                # save image, label, out
                save_image_dir = self.config.save_image_dir
                save_label_dir = self.config.save_label_dir
                save_out_dir = self.config.save_out_dir
                if not os.path.exists(save_image_dir):
                    os.makedirs(save_image_dir)
                if not os.path.exists(save_label_dir):
                    os.makedirs(save_label_dir)
                if not os.path.exists(save_out_dir):
                    os.makedirs(save_out_dir)
                save_image(input_A, os.path.join(save_image_dir, '{}_img.png'.format(i+1)),
                        normalize=True)
                save_image(label, os.path.join(save_label_dir, '{}_label.png'.format(i+1)),
                        normalize=True)
                save_image(out, os.path.join(save_out_dir, '{}_out.png'.format(i+1)),
                        normalize=True)

                print("output_png:{}   ".format((i + 1), ))
            
        
        obj = Metrix(self.config.save_label_dir, self.config.save_out_dir)
        acc, IOU, pre, rec, F1, kappa = obj.main()
        self.logger.info(
            "\nValidate set: acc: {:.4f}, IOU: {:.4f}, pre: {:.4f}, rec: {:.4f}, F1: {:.4f}, kappa: {:.4f}\n".format(
                acc, IOU, pre, rec, F1, kappa
            )
        )

    def test(self):
        self.model.eval()
        with torch.no_grad():
            print(self.data_loader.test_loader)
            for i, (input_A, label) in enumerate(self.data_loader.test_loader):
                input_A = Variable(torch.FloatTensor(input_A))
                label = Variable(torch.FloatTensor(label))
                input_A, label = input_A.cuda(), label.cuda()
                out = self.model(input_A)
                label = torch.squeeze(label, 1)
                out = torch.sigmoid(out)
                # 输出二值化的值
                # out = tensor_binary(out)
                # save image, label, out
                save_image_dir = self.config.save_image_dir
                save_label_dir = self.config.save_label_dir
                save_out_dir = self.config.save_out_dir
                if not os.path.exists(save_image_dir):
                    os.makedirs(save_image_dir)
                if not os.path.exists(save_label_dir):
                    os.makedirs(save_label_dir)
                if not os.path.exists(save_out_dir):
                    os.makedirs(save_out_dir)
                save_image(input_A, os.path.join(save_image_dir, '{}_img.png'.format(i + 1)),
                           normalize=True)
                save_image(label, os.path.join(save_label_dir, '{}_label.png'.format(i + 1)),
                           normalize=True)
                save_image(out, os.path.join(save_out_dir, '{}_out.png'.format(i + 1)),
                           normalize=True)

                print("output_png:{}   ".format((i + 1), ))

        obj = Metrix(save_label_dir, save_out_dir)
        acc, IOU, pre, rec, F1, kappa = obj.main()
        self.logger.info(
            "\nTest set: acc: {:.4f}, IOU: {:.4f}, pre: {:.4f}, rec: {:.4f}, F1: {:.4f}, kappa: {:.4f}\n".format(
                acc, IOU, pre, rec, F1, kappa
            )
        )

if __name__ == '__main__':
    x1 = Variable(torch.randn(1, 3, 224, 224))
    model = Unet()
    y = model(x1)
    print(y.shape)