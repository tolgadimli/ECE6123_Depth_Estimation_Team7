from datasets import load_dataset
import torchvision

data = load_dataset("sayakpaul/nyu_depth_v2", split="train")

resnet = torchvision.models.resnet50(pretrained=True)
vgg16 = torchvision.models.vgg16_bn(pretrained=True)
alexnet = torchvision.models.alexnet(pretrained=True)

