
import argparse

parser = argparse.ArgumentParser(description="Unet3D")

parser.add_argument('--n_threads', type=int, default=8, help='number of threads for data loading')
parser.add_argument('--cpu', type=bool, default=False, help='use cpu to train')
parser.add_argument('--gpu_id', type=list, default=[0], help='use gpu')

parser.add_argument('--n_labels', type=int, default=3, help='number of classes') #分割直肠为2，分割直肠和肿瘤为3
parser.add_argument('--dataset_path', default='../train', help='trainset root path')
parser.add_argument('--testset_path', default='../test', help='testset root path')
parser.add_argument('--save', default='../files', help='save path of trained model')
parser.add_argument('--batch_size', type=int, default=2, help='batch size of trainset')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--epochs', type=int, default=500, help='number of epochs to train')

args = parser.parse_args()