import torch
from config import DefaultConfig
import torchvision
from torchvision import transforms, utils
from Model import CRACK_Model1
import matplotlib.pyplot as plt

# 图像增光
transform = transforms.Compose(
    [transforms.ToTensor(),
    # transforms.RandomRotation(5),
     transforms.Normalize((0.5,), (0.5,))])

opt = DefaultConfig()
use_gpu = opt.use_gpu


def test(testloader, model):
    with torch.no_grad():
        tp_sum = 0
        fp_sum = 0
        fn_sum = 0

        for i, data in enumerate(testloader, 0):
            input_img, labels = data
            # print(input_img.size())
            if use_gpu:
                input_img = input_img.cuda()
                labels = labels.cuda()
            output = model(input_img)
            _, predict = torch.max(output, 1)

            tp = (predict[labels == 1] == 1).float()
            fp = (predict[labels == 0] == 1).float()
            fn = (predict[labels == 1] == 0).float()

            tp_sum += tp.sum()
            fp_sum += fp.sum()
            fn_sum += fn.sum()

        prec = tp_sum / (tp_sum + fp_sum + 1e-6)
        recall = tp_sum / (fn_sum + tp_sum + 1e-6)
        f1_score = 2 * (prec * recall) / (recall + prec + 1e-6)

        return prec, recall, f1_score

model = CRACK_Model1()

testset = torchvision.datasets.ImageFolder("./test", transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False)

model.load_state_dict(torch.load('./model.pt', map_location=torch.device('cpu')))

if use_gpu: model = model.cuda()

prec, recall, f1_score = test(testloader, model)

print(prec, recall, f1_score)