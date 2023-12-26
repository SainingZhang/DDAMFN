import os
import sys
import glob
from tqdm import tqdm
import argparse

from PIL import Image
import numpy as np

import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.utils.data as data
from networks.DDAMNet import DDAMNet
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools

eps = sys.float_info.epsilon
from thop import profile
from torchstat import stat
from collections import OrderedDict
from torchsummary import summary
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--aff_path', type=str, default='/data/dataset/fer/fer_112_112_data/open_data/affectnet/affectnet_train_val_org_data/', help='AfectNet dataset path.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate for adam.')
    parser.add_argument('--workers', default=8, type=int, help='Number of data loading workers.')
    parser.add_argument('--epochs', type=int, default=40, help='Total training epochs.')
    parser.add_argument('--num_head', type=int, default=2, help='Number of attention head.')
    parser.add_argument('--num_class', type=int, default=7, help='Number of class.')
    parser.add_argument('--model_path', default = '/data/2021/code/fer/paper_2021/DAN-main/DAN-main_new/DDAM_main/checkpoints/affectnet7_best/affecnet7_epoch10_acc0.67.pth')
    return parser.parse_args()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
   # fmt = '%0.2f%%' if normalize else 'd'
    
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j]*100, fmt)+'%',
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")


    plt.ylabel('Actual', fontsize=18)
    plt.xlabel('Predicted', fontsize=18)
    plt.tight_layout()


class7_names = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Angry'] 
class8_names = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Angry', 'Contempt'] 

def run_test():
    args = parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = DDAMNet(num_class=args.num_class, num_head=args.num_head, pretrained=False)
    #stat(model, (3, 112, 112))    
    checkpoint = torch.load(args.model_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()        
  #  summary(model, input_size=(3,112,112), batch_size=-1, device='cuda')

    data_transforms_val = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])     
                                                                      
 
    val_dataset = datasets.ImageFolder(f'{args.aff_path}/val', transform = data_transforms_val)    # loading statically
    if args.num_class == 7:   # ignore the 8-th class 
        idx = [i for i in range(len(val_dataset)) if val_dataset.imgs[i][1] != 7]
        val_dataset = data.Subset(val_dataset, idx)

    print('Validation set size:', val_dataset.__len__())
    
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                               batch_size = args.batch_size,
                                               num_workers = args.workers,
                                               shuffle = False,  
                                               pin_memory = True)

    iter_cnt = 0
    bingo_cnt = 0
    sample_cnt = 0
    
    for imgs, targets in val_loader:
    
        imgs = imgs.to(device)
        targets = targets.to(device)
        out,feat,heads = model(imgs)


        _, predicts = torch.max(out, 1)
        correct_num  = torch.eq(predicts,targets)
        bingo_cnt += correct_num.sum().cpu()
        sample_cnt += out.size(0)
        
        if iter_cnt == 0:
            all_predicted = predicts
            all_targets = targets
        else:
            all_predicted = torch.cat((all_predicted, predicts),0)
            all_targets = torch.cat((all_targets, targets),0)                  
        iter_cnt+=1        


    acc = bingo_cnt.float()/float(sample_cnt)
    acc = np.around(acc.numpy(),4)

    tqdm.write("Validation accuracy:%.4f. " % ( acc))


    if args.num_class == 7:
        torch.save({'model_state_dict': model.state_dict()}, os.path.join('checkpoints', "affecnet7_epoch10_test"+"_acc"+str(acc)+".pth"))
        tqdm.write('Model saved.')    
        # Compute confusion matrix
        matrix = confusion_matrix(all_targets.data.cpu().numpy(), all_predicted.cpu().numpy())
        np.set_printoptions(precision=2)
        plt.figure(figsize=(10, 8))
        # Plot normalized confusion matrix
        plot_confusion_matrix(matrix, classes=class7_names, normalize=True, title= 'affectnet  Confusion Matrix (acc: %0.2f%%)' %(acc*100))
 		
        plt.savefig(os.path.join('checkpoints', "affecnet7_epoch10"+"_acc"+str(acc)+".png"))
        plt.close()				

    elif args.num_class == 8:
        matrix = confusion_matrix(all_targets.data.cpu().numpy(), all_predicted.cpu().numpy())
        np.set_printoptions(precision=2)
        plt.figure(figsize=(10, 8))
        plot_confusion_matrix(matrix, classes=class8_names, normalize=True, title= 'raf Confusion Matrix (acc: %0.2f%%)' %(acc*100))
 		
        plt.savefig(os.path.join('checkpoints', "affecnet8_epoch"+"_acc"+str(acc)+".png"))
        plt.close()	

        
if __name__ == "__main__":                    
    run_test()