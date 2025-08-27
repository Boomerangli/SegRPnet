import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import glob
import os

class Metrix:
    def __init__(self, label_folder, output_folder):
        self.label_folder = label_folder
        self.label_Imgs = sorted(glob.glob(self.label_folder + '/*.*'))
        self.output_folder = output_folder
        self.output_Imgs = sorted(glob.glob(self.output_folder + '/*.*'))

    def get_metric(self,Output, GroundTruth):
        Output = Output
        GroundTruth = GroundTruth
        TP = (Output > 0.5) & (GroundTruth > 0.5)
        FP = (Output > 0.5) & (GroundTruth <= 0.5)
        FN = (Output <= 0.5) & (GroundTruth > 0.5)
        TN = (Output <= 0.5) & (GroundTruth <= 0.5)

        TP = float(torch.sum(TP))
        FP = float(torch.sum(FP))
        FN = float(torch.sum(FN))
        TN = float(torch.sum(TN))

        return TP, FP, FN, TN

    def LoadImg(self,imgpath):
        img = Image.open(imgpath)
        img = img.convert('L')
        threshold = 128  #
        table = []
        for i in range(256):
            if i < threshold:
                table.append(0)
            else:
                table.append(1)
        # 根据阈值二值化
        img = img.point(table, '1')
        img = transforms.ToTensor()(img)
        return img

    def cal_metrics_value(self,TP, FP, FN, TN):
        acc1 = TP + TN
        acc2 = TP + TN + FP + FN
        acc = acc1 / acc2 if acc2 > 0.001 else 0.0

        IOU_A = TP + FP + FN
        IOU = TP / IOU_A if IOU_A > 0.001 else 0.0

        pr1 = TP + FP
        pre = TP / pr1 if pr1 > 0.001 else 0.0

        rec1 = TP + FN
        rec = TP / rec1 if rec1 > 0.001 else 0.0

        F1 = 2 * rec * pre / (rec + pre + 1e-6) if (rec + pre + 1e-6) > 0.001 else 0.0

        P0 = acc
        Pe_1 = (TP + FN) * (TP + FP) + (FP + TN) * (FN + TN)
        Pe = Pe_1 / acc2 ** 2 if acc2 ** 2 > 0.001 else 0.0
        kappa = (P0 - Pe) / (1 - Pe) if (1 - Pe) > 0.001 else 0.0

        return acc, IOU, pre, rec, F1, kappa

    def del_all_files(self):
        print("delete all files in 2 directory")
        dir_list = []
        # dir_list.append(self.orignal_img_A_folder)
        # dir_list.append(self.orignal_img_B_folder)
        dir_list.append(self.label_folder)
        dir_list.append(self.output_folder)
        for path in dir_list:
            for i in os.listdir(path):
                path_file = os.path.join(path, i)
                if os.path.isfile(path_file):
                    os.remove(path_file)
        isNull_flag = []
        for path in dir_list:
            isNull_flag.append(os.path.exists(path))
        if (isNull_flag[0] == isNull_flag[1]) and (isNull_flag[1] == isNull_flag[2]):
            print("模型验证时产生的测试图片已经清空")
        else:
            print("校验测试文件夹未清空")


    def main(self):
        TP_list = []
        FP_list = []
        FN_list = []
        TN_list = []
        assert len(self.output_Imgs) == len(self.label_Imgs)
        for i in range(len(self.output_Imgs)):
            output = self.LoadImg(self.output_Imgs[i])
            label = self.LoadImg(self.label_Imgs[i])

            TP, FP, FN, TN = self.get_metric(output, label)
            TP_list.append(TP)
            FP_list.append(FP)
            FN_list.append(FN)
            TN_list.append(TN)
            # 计算单张影像精度指标
            acc, IOU, pre, rec, F1, kappa = self.cal_metrics_value(TP, FP, FN, TN)
           # print(i, )
           # print('精度指标：', IOU, )
           # print(i)

        TP_SUM = np.sum(TP_list)
        FP_SUM = np.sum(FP_list)
        TN_SUM = np.sum(TN_list)
        FN_SUM = np.sum(FN_list)

        # print('数据集直接裁切')
        print('总数')
        print('TP', TP_SUM)
        print('TN', TN_SUM)
        print('FN', FN_SUM)
        print('FP', FP_SUM)
        # print('比值',TN_SUM/TP_SUM)
        print('*******************')
        print(' 指标计算 :')

        IOU_A = TP_SUM + FP_SUM + FN_SUM
        IOU = TP_SUM / IOU_A
        IOU = 100 * IOU
        print('      IoU:', format(IOU, '.2f'))

        acc1 = TP_SUM + TN_SUM
        acc2 = TP_SUM + TN_SUM + FP_SUM + FN_SUM
        acc = acc1 / acc2
        acc = acc * 100
        print(' Accuracy:', format(acc, '.2f'))

        rec1 = TP_SUM + FN_SUM
        rec = TP_SUM / rec1
        rec = 100 * rec
        print('   recall:', format(rec, '.2f'))

        pr1 = TP_SUM + FP_SUM
        pre = TP_SUM / pr1
        pre = 100 * pre
        print('Precision:', format(pre, '.2f'))

        F1 = 2 * rec * pre / (rec + pre + 1e-6)
        print('       F1:', format(F1, '.2f'))
        #

        P0 = acc1 / acc2
        Pe_1 = (TP_SUM + FN_SUM) * (TP_SUM + FP_SUM) + (FP_SUM + TN_SUM) * (FN_SUM + TN_SUM)
        Pe = Pe_1 / acc2 ** 2
        kappa = (P0 - Pe) / (1 - Pe)
        kappa = 100 * kappa
        print('    kappa:', format(kappa, '.2f'))
        # self.del_all_files()

        return acc, IOU, pre, rec, F1, kappa


if __name__ == '__main__':
    label_folder = r'E:\WeaklyCD\PTH\CD\SWCD_pth\Unet+CBAM\label'
    output_folder = r'E:\WeaklyCD\PTH\CD\SWCD_pth\Unet+CBAM\out'
    obj = Metrix(label_folder, output_folder)
    result = obj.main()




