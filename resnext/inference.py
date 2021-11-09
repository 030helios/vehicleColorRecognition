import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import resnext101_32x8d

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # create model
    model = resnext101_32x8d(num_classes=200).to(device)

    # load model weights
    weights_path = "./resnext101.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    # prediction
    model.eval()

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # load image names
    data_root = os.path.abspath(os.path.join(
        os.getcwd(), os.pardir))  # get data root path
    assert os.path.exists(data_root), "{} path does not exist.".format(data_root)
    training_images = os.path.join(data_root, "testing_images")
    with open(training_images+'/testing_img_order.txt') as f:
        lines = f.read().splitlines()

    outputfile = open('answer.txt', 'w')

    # load image
    img_path_list = []
    img_list = []
    for line in lines:
        img_path = training_images+"/"+line
        assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
        img = Image.open(img_path)
        img = data_transform(img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)

        with torch.no_grad():
            # predict class
            output = torch.squeeze(model(img.to(device))).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()

        print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                     predict[predict_cla].numpy())
        outputfile.write(line)
        outputfile.write(" ")
        outputfile.write(class_indict[str(predict_cla)])
        outputfile.write("\n")

if __name__ == '__main__':
    main()
