"""
[NOTE] Using command: "pip install grad-cam==1.3.1"
to install the pytorch_grad_cam package
"""

import os
import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models

from pytorch_grad_cam import GradCAM, \
                             ScoreCAM, \
                             GradCAMPlusPlus, \
                             AblationCAM, \
                             XGradCAM, \
                             EigenCAM, \
                             EigenGradCAM

from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
                                         deprocess_image, \
                                         preprocess_image

#from networks.dan import DAN
#from networks.dacl import resnet18
from networks.DDAMNet import DDAMNet



if __name__ == '__main__':

    os.makedirs('cam_affect7_result', exist_ok=True)

    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,}

    ## visualize by DACL model
    # model = DaclModel()
    # names = {
    #     'baseline':model.model.layer4[-1]
    # }

    ## visualize by DAN model
    #model = DAN(num_class=7, num_head=4)   
    model = DDAMNet(num_class=7, num_head=2)
    #model = mobilenet(num_class=args.num_class, num_head=args.num_head) 
    
    checkpoint = torch.load('/data/2021/code/fer/paper_2021/DAN-main/DAN-main_new/DDAM_main/checkpoints/affectnet7_best/affecnet7_epoch10_acc0.67.pth')
    model.load_state_dict(checkpoint['model_state_dict'],strict=True) 
    names = {
       # 'our_head0':model.cat_head0.CoordAtt,
       # 'our_head1':model.cat_head1.CoordAtt,
        #'our_head2':model.cat_head2.sa,
        #'our_head3':model.cat_head3.sa,
        # 'our_head4':model.cat_head4.sa,
        'our_head1':model.features[-1],
    }
    #target_layers = [model.features[-1].norm1]
    ## select part of test data to gen
    #for p in glob.glob('/data/2021/code/fer/paper_2021/DAN-main/DAN-main_new/DAN-main/datasets/test/test*.jpg')[100:300]:
    for p in glob.glob('/data/2021/code/fer/paper_2021/DAN-main/DAN-main_new/DDAM_main/test_data/*.jpg')[:]:
        for name,target_layer in names.items():
            cam = methods['gradcam++'](model=model,
                            target_layer=target_layer,
                            use_cuda=True)

            rgb_img = cv2.imread(p, 1)[:, :, ::-1]
            rgb_img = np.float32(rgb_img) / 255
            input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406], 
                                                    std=[0.229, 0.224, 0.225])

            target_category = None
            cam.batch_size = 32

            grayscale_cam = cam(input_tensor=input_tensor,
                                target_category=target_category,
                                aug_smooth=False,
                                eigen_smooth=False)

            # Here grayscale_cam has only one image in the batch
#            grayscale_cam = grayscale_cam[0, :]
            grayscale_cam = grayscale_cam[0]

            cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

            # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
            cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

            gb_model = GuidedBackpropReLUModel(model=model, use_cuda=True)
            gb = gb_model(input_tensor, target_category=target_category)

            cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
            cam_gb = deprocess_image(cam_mask * gb)
            gb = deprocess_image(gb)

            cv2.imwrite(f'./cam_affect7_result/{os.path.basename(p)}_{name}_cam.jpg', cam_image)