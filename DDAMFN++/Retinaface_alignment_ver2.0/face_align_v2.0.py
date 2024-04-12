# _*_ coding:utf-8 _*_
import os
import cv2
import numpy

import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
import shutil
import random
from skimage import transform as trans
parser = argparse.ArgumentParser(description='Retinaface')

parser.add_argument('-m', '--trained_model', default='./weights/mobilenet0.25_Final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='mobile0.25', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--confidence_threshold', default=0.6, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=1, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.35, type=float, help='nms_threshold')
parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')

args = parser.parse_args()

dataset_folder = './data/fer_org/rafdb/train/'
dst_folder = "./data/fer_112_112_v2.0/rafdb/train/"

#-->left profile
src1 = np.array([[55.550, 49.477], [62.116, 49.339],  [38.075, 70.237], [55.017, 92.262], [61.466, 92.979]], dtype=np.float32)
#<--left                
src2 = np.array([[47.241, 48.598], [72.887, 49.540],  [40.555, 71.067], [47.423, 93.643], [71.236, 94.353]], dtype=np.float32)
#---frontal                                                                                               
src3 = np.array([[33.566, 40.559], [78.321, 40.559],  [55.338, 64.967], [38.479, 90.276], [73.591, 90.276]], dtype=np.float32)
#-->right
src4 = np.array([[39.516, 49.540], [65.162, 48.598],  [71.849, 71.067], [41.167, 94.353], [64.980, 93.643]], dtype=np.float32)
#-->right profile
src5 = np.array([[50.225, 49.339], [56.791, 49.477],  [74.266, 70.237], [50.875, 92.979], [65.317, 92.262]], dtype=np.float32)

src = np.array([src1, src2, src3, src4, src5])
src_map = {112: src, 224: src * 2} 

# lmk is prediction; src is template
def estimate_norm(lmk, image_size=112):
    assert lmk.shape == (5, 2)
    tform = trans.SimilarityTransform()
    lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1)
    min_M = []
    min_index = []
    min_error = float('inf')

    src = src_map[image_size]
    for i in np.arange(src.shape[0]):
        tform.estimate(lmk, src[i])
        M = tform.params[0:2, :]
        results = np.dot(M, lmk_tran.T)
        results = results.T
        error = np.sum(np.sqrt(np.sum((results - src[i])**2, axis=1)))
        #         print(error)
        if error < min_error:
            min_error = error
            min_M = M
            min_index = i
    return min_M, min_index

def norm_crop(img, landmark, image_size=112):
  M, pose_index = estimate_norm(landmark, image_size)
  warped = cv2.warpAffine(img,M, (image_size, image_size), borderValue = 0.0)
  return warped

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True

def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

if __name__ == '__main__':
    torch.set_grad_enabled(False)
    cfg = None
    if args.network == "mobile0.25":
        cfg = cfg_mnet
    elif args.network == "resnet50":
        cfg = cfg_re50
    # net and model
    net = RetinaFace(cfg=cfg, phase = 'test')
    net = load_model(net, args.trained_model, args.cpu)
    net.eval()
    print('Finished loading model!')
    print(net)
    cudnn.benchmark = True
    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)

    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    # testing scale
    resize = 1
    count = 0

    for sub_folder in os.listdir(dataset_folder):
        sub_dir = os.path.join(dataset_folder, sub_folder)

        if (os.path.isfile(sub_dir)):
            continue
        
        for sub_folder_1 in os.listdir(sub_dir):
            sub_dir_1 = os.path.join(sub_dir, sub_folder_1)
            if (os.path.isfile(sub_dir_1)):
                continue
            files = os.listdir(sub_dir_1)
            
            for img_file_name in files:            
                relative_path = os.path.join(sub_dir_1, img_file_name)
                file_relative_path = os.path.join(sub_folder_1, img_file_name)
            
                org_img_file_name = img_file_name            
            
                dst_sub_folder = os.path.join(dst_folder, sub_folder_1)
                if not os.path.exists(dst_sub_folder):
                    os.makedirs(dst_sub_folder)

                if relative_path.endswith(('jpg','JPG','png','jpeg','bmp','PNG','JPEG')):
                    dst_image_folder =os.path.join(dst_folder, sub_folder_1)
                    if not os.path.exists(dst_image_folder):
                        os.makedirs(dst_image_folder)                
                    org_img = cv2.imread(relative_path, cv2.IMREAD_COLOR)
                    img_raw = org_img           
            
                    img = np.float32(img_raw)
                    try:
                        org_height = img_raw.shape[0]
                        org_width = img_raw.shape[1]
                    except:
                        print(relative_path)
                        continue            
            
                    # testing scale
                    long_size = 320
                    im_shape = img.shape
                    im_size_min = np.min(im_shape[0:2])
                    im_size_max = np.max(im_shape[0:2])
                    resize = float(long_size) / float(im_size_min)
                    if np.round(resize * im_size_max) > long_size:
                        resize = float(long_size) / float(im_size_max)
            
                    if resize != 1:
                        img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
                    im_height = img.shape[0]
                    im_width = img.shape[1]
                    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
                    img -= (104, 117, 123)
                    img = img.transpose(2, 0, 1)
                    img = torch.from_numpy(img).unsqueeze(0)
                    img = img.to(device)
                    scale = scale.to(device)
            
                    loc, conf, landms = net(img)  # forward pass

                    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
                    prior_data = priorbox.forward().to(device).data

                    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
                    boxes = boxes * scale / resize
                    boxes = boxes.cpu().numpy()
                    
                    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
                    
                    landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
                    
                    scale1_size = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                   img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                                   img.shape[3], img.shape[2]]).to(device)

                    landms = landms * scale1_size / resize
                    landms = landms.cpu().numpy()
            
                    # ignore low scores
                    inds = np.where(scores > args.confidence_threshold)[0]
                    boxes = boxes[inds]
                    landms = landms[inds]
                    scores = scores[inds]
            
                  # keep top-K before NMS
                    order = scores.argsort()[::-1][:args.top_k]
                    boxes = boxes[order]
                    landms = landms[order]
                    scores = scores[order]
            
                   # do NMS
                    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
                    keep = py_cpu_nms(dets, args.nms_threshold)
                    dets = dets[keep, :]
                    landms = landms[keep]
            
                    dets = np.concatenate((dets, landms), axis=1)
                    if(dets.shape[0] <= 0):
                        print('not face in '+relative_path)
                        crop_im2 = cv2.resize(org_img, (112, 112), interpolation=cv2.INTER_LINEAR)                        
                        dst_file_name = os.path.splitext(org_img_file_name)[0]
                        dst_img_file = os.path.join(dst_image_folder,dst_file_name+'.jpg')
                        cv2.imwrite(dst_img_file, crop_im2)
                        count = count + 1
                        continue 
                    
                    for b in dets:
                        if b[4] < args.vis_thres:
                            continue
                       # text = "{:.4f}".format(b[4])
                        b = list(map(int, b))
            
                        left_eye_x = b[5]
                        right_eye_x = b[7]
                        nose_x = b[9]
                        left_mouth_x = b[11]
                        right_mouth_x = b[13]
                        left_eye_y = b[6]
                        right_eye_y = b[8]
                        nose_y = b[10]
                        left_mouth_y = b[12]
                        right_mouth_y = b[14]
                                          
                        face_landmarks = np.array([[left_eye_x,left_eye_y], 
                                          [right_eye_x,right_eye_y],
                                          [nose_x,nose_y],
                                          [left_mouth_x,left_mouth_y],
                                          [right_mouth_x,right_mouth_y] ], dtype=np.float32 )                                     
            
                        dst2  = norm_crop(org_img,face_landmarks)
                        
                        dst_file_name = os.path.splitext(org_img_file_name)[0]
                        dst_img_file = os.path.join(dst_image_folder,dst_file_name+'.jpg')
 
                        cv2.imwrite(dst_img_file, dst2)

                        count = count + 1
                        break                    
print('done')
print(count)  
