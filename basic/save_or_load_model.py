import torch
import torch.onnx as onnx
import torchvision.models as models

model = models.vgg16(pretrained=True)
# PyTorch 模型将学习到的参数存储在内部状态字典中，称为state_dict
torch.save(model.state_dict(), 'data/model_weights.pth')

model = models.vgg16()  # we do not specify pretrained=True, i.e. do not load default weights
model.load_state_dict(torch.load('data/model_weights.pth'))
model.eval()

torch.save(model, 'data/vgg_model.pth')
model = torch.load('data/vgg_model.pth')

input_image = torch.zeros((1,3,224,224))
onnx.export(model, input_image, 'data/model.onnx')