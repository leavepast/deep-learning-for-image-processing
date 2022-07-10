from vit_model import vit_base_patch16_224 as create_model
import hiddenlayer as h
import torch
model = create_model(num_classes=5)
vis_graph = h.build_graph(model, torch.zeros([1 ,3, 224, 224]))   # 获取绘制图像的对象
vis_graph.theme = h.graph.THEMES["blue"].copy()     # 指定主题颜色
vis_graph.save("./demo1")   # 保存图像的路径
#
# from torchviz import make_dot
# x = torch.randn(1, 3, 224, 224).requires_grad_(True)  # 定义一个网络的输入值
# y = model(x)    # 获取网络的预测值
# MyConvNetVis = make_dot(y, params=dict(list(model.named_parameters()) + [('x', x)]))
# #MyConvNetVis.format = "png"
# # 指定文件生成的文件夹
# MyConvNetVis.directory = "data"
# # 生成文件
# MyConvNetVis.view()