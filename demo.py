import os
import time
import math

import torch
import torchvision.transforms as transforms
from PIL import Image

import utils
from model import SVTR


# DEVICE
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# MODEL
list_of_characters = """aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ0123456789!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~ îç¤²€û°ëœ“”Ō…ÖĀ’–öü—−ŪōÜāūÐð™"""

converter = utils.CTCLabelConverter(list_of_characters)

num_class = len(converter.character)
input_channel = 1
model = SVTR(num_classes=num_class)
model.load_state_dict(torch.load('pretrained_model/best_norm_ED_final.pth'))
model.to(device)
model.eval()

# DATA
class NormalizePAD(object):

    def __init__(self, max_size, PAD_type='right'):
        self.toTensor = transforms.ToTensor()
        self.max_size = max_size
        self.max_width_half = math.floor(max_size[2] / 2)
        self.PAD_type = PAD_type

    def __call__(self, img):
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        c, h, w = img.size()
        Pad_img = torch.FloatTensor(*self.max_size).fill_(0)
        Pad_img[:, :, :w] = img  # right pad
        if self.max_size[2] != w:  # add border Pad
            Pad_img[:, :, w:] = img[:, :, w - 1].unsqueeze(2).expand(c, h, self.max_size[2] - w)

        return Pad_img

class Align(object):

    def __init__(self, input_channel=1, imgH=32, imgW=1024):
        self.imgH = imgH
        self.imgW = imgW
        self.transform = NormalizePAD((input_channel, self.imgH, self.imgW))

    def __call__(self, image):
        w, h = image.size
        ratio = w / float(h)
        if math.ceil(self.imgH * ratio) > self.imgW:
            resized_w = self.imgW
        else:
            resized_w = math.ceil(self.imgH * ratio)
        
        resized_image = image.resize((resized_w, self.imgH), Image.BICUBIC)
        resized_image = self.transform(resized_image)

        return resized_image.unsqueeze(0)


transforms = Align()

path_test = 'data/test'
list_image = os.listdir(path_test)

# PREDICT
sum_inference_time = 0
with torch.no_grad():
    for image_path in list_image:
        img_path = os.path.join(path_test, image_path)
        img = Image.open(img_path).convert('L')
        image_tensors = transforms(img)
        batch_size = image_tensors.size(0)
        image_tensors = image_tensors.to(device)

        t0 = time.time()
        preds = model(image_tensors)
        t1 = time.time()
        sum_inference_time += (t1-t0)

        # Select max probabilty (greedy decoding) then decode index to character
        preds_size = torch.IntTensor([preds.size(1)] * batch_size)
        _, preds_index = preds.max(2)
        preds_str = converter.decode(preds_index, preds_size)
        print("\nImage: ", image_path)
        print("Predict: ", preds_str)

time_one_image = sum_inference_time/len(list_image)
print('Time cost for one image: ', time_one_image)
fps = float(1/time_one_image)
print("FPS = {} ".format(fps, '.1f') )

