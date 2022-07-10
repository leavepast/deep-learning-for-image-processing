from vit_model import vit_base_patch16_224 as create_model
import torch
from collections import OrderedDict
model=create_model(patch_size=1,num_classes=28)
inputs = torch.randn(1, 3, 224,224)
weights_res50_vit=torch.load('R50+ViT-B_16_imagenet21k_imagenet2012.pth')
weights_vit=torch.load('/media/xjw/doc/00-ubuntu-files/vit/model/vit_base_patch16_224.pth')
weights_res50_vit = OrderedDict(weights_res50_vit)
for name in weights_res50_vit.keys():
    if  name.startswith("resnet"):
        weights_vit[name]=weights_res50_vit[name]
pass