'''
@paper: GAN Prior Embedded Network for Blind Face Restoration in the Wild (CVPR2021)
@author: yangxy (yangtao9009@gmail.com)
'''
import torch
import os
import cv2
import glob
import numpy as np
from torch import nn
import torch.nn.functional as F
from torchvision import transforms, utils
from .model import FullGenerator
from torch.autograd import Variable
import torch.onnx
import onnx
import onnxruntime
from PIL import Image


class FaceGAN(object):
    def __init__(self, base_dir=None, size=512, model=None, channel_multiplier=2, narrow=1, is_norm=True, use_cpu=False, use_onnx=False):
        if base_dir:
            self.mfile = os.path.join(base_dir, 'weights', model+'.pth')
            self.onnxfile = os.path.join(base_dir, 'weights', model+'.onnx')
        else:
            base = os.path.dirname(os.path.realpath(__file__))
            self.mfile = os.path.join(base, '..', 'weights', model+'.pth')
            self.onnxfile = os.path.join(base, '..', 'weights', model+'.onnx')
        
        if use_cpu:
            torch.cuda.is_available = lambda : False
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.use_cpu = use_cpu        
        self.use_onnx = use_onnx        
            
        self.n_mlp = 8
        self.is_norm = is_norm
        self.resolution = size
        self.load_model(channel_multiplier, narrow)
        

    def load_model(self, channel_multiplier=2, narrow=1):
    
        if self.use_onnx:
            so = onnxruntime.SessionOptions()
            so.log_severity_level = 3
            print(onnxruntime.get_device())
            self.ort_session = onnxruntime.InferenceSession(self.onnxfile, so)
            
        else:
            self.model = FullGenerator(self.resolution, 512, self.n_mlp, channel_multiplier, narrow=narrow).to(self.device)
            pretrained_dict = torch.load(self.mfile, map_location=self.device)
            self.model.load_state_dict(pretrained_dict)
            self.model.eval()
        
    def export(self, name):
        dummy_input = Variable(torch.randn(1, 3, 512, 512))
        torch.onnx.export(self.model, dummy_input, name+".onnx", opset_version=10,verbose=True, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)

    def process(self, img, output_size=None):
        img = cv2.resize(img, (self.resolution, self.resolution))
        img_t = self.img2tensor(img)
        
        def to_numpy(tensor):
            return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
        
        if self.use_onnx:
            ort_inputs = {self.ort_session.get_inputs()[0].name: to_numpy(img_t) }
            ort_outs = self.ort_session.run(None, ort_inputs)
            out = torch.tensor(ort_outs[0]).cpu()

        else:
            with torch.no_grad():
                out, __ = self.model(img_t)
        
        out = self.tensor2img(out)
        if output_size is not None:
            out = cv2.resize(out, (output_size, output_size))
        
        return out
        
    

    def img2tensor(self, img):

        img_t = torch.from_numpy(img).to(self.device)/255.

        if self.is_norm:
            img_t = (img_t - 0.5) / 0.5
        img_t = img_t.permute(2, 0, 1).unsqueeze(0).flip(1) # BGR->RGB
        return img_t

    def tensor2img(self, img_t, pmax=255.0, imtype=np.uint8):
        if self.is_norm:
            img_t = img_t * 0.5 + 0.5
        img_t = img_t.squeeze(0).permute(1, 2, 0).flip(2) # RGB->BGR
        img_np = np.clip(img_t.float().cpu().numpy(), 0, 1) * pmax

        return img_np.astype(imtype)
