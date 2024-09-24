import os
import time
from dataclasses import dataclass

import torch
from torch.nn import init
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
import numpy as np
from nltk.metrics.distance import edit_distance

from model import SVTR
import dataset
import utils


# DEVICE
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# DATASETS
list_of_characters = """aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ0123456789!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~ îç¤²€û°ëœ“”Ō…ÖĀ’–öü—−ŪōÜāūÐð™"""

@dataclass
class Config:
    character: str = list_of_characters
    batch_max_length: int = 128
    rgb: bool = False
    imgW: int = 1024
    imgH: int = 32
    keep_ratio_with_pad: bool = True

opt = Config()

train_data = 'data/trainset'
train_dataset = dataset.LmdbDataset(root=train_data, opt=opt)
train_loader = DataLoader(train_dataset, 
                          batch_size=2, 
                          shuffle=True, 
                          collate_fn=dataset.AlignCollate(opt.imgH, opt.imgW, opt.keep_ratio_with_pad))

val_data = 'data/valset'
val_dataset = dataset.LmdbDataset(root=val_data, opt=opt)
val_loader = DataLoader(val_dataset, 
                        batch_size=1, 
                        shuffle=True, 
                        collate_fn=dataset.AlignCollate(opt.imgH, opt.imgW, opt.keep_ratio_with_pad))

print(f"Length of Train Dataset: {len(train_dataset)}")
print(f"Length of Validation Dataset: {len(val_dataset)}")

# check data loaders
batch = next(iter(train_loader))
print(len(batch))

# MODEL
converter = utils.CTCLabelConverter(opt.character)

opt.num_class = len(converter.character)
opt.input_channel = 3 if opt.rgb else 1
model = SVTR(num_classes=opt.num_class, img_size=(opt.imgH, opt.imgW), input_channel=opt.input_channel, max_seq_len=opt.imgW//4)
print(model)
"""
# weight initialization
for name, param in model.named_parameters():
    try:
        if 'bias' in name:
            init.constant_(param, 0.0)
        elif 'weight' in name:
            init.kaiming_normal_(param)
    except Exception as e:  # for BatchNorm.
        if 'weight' in name:
            param.data.fill_(1)
        continue
"""
# data parallel for multi-GPU
# model = torch.nn.DataParallel(model).to(device)
model.to(device)
model.train()

# LOSS
criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
# loss averager
loss_avg = utils.Averager()

# OPTIMIZER
# filter that only require gradient decent
filtered_parameters = []
params_num = []
for p in filter(lambda p: p.requires_grad, model.parameters()):
    filtered_parameters.append(p)
    params_num.append(np.prod(p.size()))
print('Trainable params num : ', sum(params_num))

# setup optimizer
optimizer = optim.AdamW(filtered_parameters, lr=0.001)
print("Optimizer:")
print(optimizer)

# TRAIN
def validation(model, criterion, evaluation_loader, converter, opt):
    """ validation or evaluation """
    n_correct = 0
    norm_ED = 0
    length_of_data = 0
    infer_time = 0
    valid_loss_avg = utils.Averager()

    for i, (image_tensors, labels) in enumerate(evaluation_loader):
        batch_size = image_tensors.size(0)
        length_of_data = length_of_data + batch_size
        image = image_tensors.to(device)

        text_for_loss, length_for_loss = converter.encode(labels, batch_max_length=opt.batch_max_length)
        
        start_time = time.time()
        preds = model(image)
        forward_time = time.time() - start_time

        # Calculate evaluation loss for CTC deocder.
        preds_size = torch.IntTensor([preds.size(1)] * batch_size)
        # permute 'preds' to use CTCloss format
        cost = criterion(preds.log_softmax(2).permute(1, 0, 2), text_for_loss, preds_size, length_for_loss)

        # Select max probabilty (greedy decoding) then decode index to character
        _, preds_index = preds.max(2)
        preds_str = converter.decode(preds_index.data, preds_size.data)

        infer_time += forward_time
        valid_loss_avg.add(cost)

        # calculate accuracy & confidence score
        preds_prob = F.softmax(preds, dim=2)
        preds_max_prob, _ = preds_prob.max(dim=2)
        confidence_score_list = []
        for gt, pred, pred_max_prob in zip(labels, preds_str, preds_max_prob):
            if pred == gt:
                n_correct += 1
            
            # ICDAR2019 Normalized Edit Distance
            if len(gt) == 0 or len(pred) == 0:
                norm_ED += 0
            elif len(gt) > len(pred):
                norm_ED += 1 - edit_distance(pred, gt) / len(gt)
            else:
                norm_ED += 1 - edit_distance(pred, gt) / len(pred)

            # calculate confidence score (= multiply of pred_max_prob)
            try:
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]
            except:
                confidence_score = 0  # for empty pred case, when prune after "end of sentence" token ([s])
            confidence_score_list.append(confidence_score)

    accuracy = n_correct / float(length_of_data) * 100
    norm_ED = norm_ED / float(length_of_data)  # ICDAR2019 Normalized Edit Distance

    return valid_loss_avg.val(), accuracy, norm_ED, preds_str, confidence_score_list, labels, infer_time, length_of_data

best_accuracy = -1
best_norm_ED = -1
opt.grad_clip = 5
epochs = 25

for epoch in range(epochs):
    print('--------------------------------------------------------')
    print('Epoch: ', epoch)
    start_time = time.time()

    # Train part
    for i, (image_tensors, labels) in enumerate(train_loader):
        image = image_tensors.to(device)
        text, length = converter.encode(labels, batch_max_length=opt.batch_max_length)
        batch_size = image.size(0)

        # zero the parameter gradients
        optimizer.zero_grad()

        preds = model(image)
        preds_size = torch.IntTensor([preds.size(1)] * batch_size)
        preds = preds.log_softmax(2).permute(1, 0, 2)
        cost = criterion(preds, text, preds_size, length)

        cost.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)  # gradient clipping with 5 (Default)
        optimizer.step()

        loss_avg.add(cost)

        if i % 1000 == 0:
            print(f'Batch {i}/{len(train_loader)}, Train loss: {loss_avg.val()}')
    
    # Validation part
    model.eval()
    with torch.no_grad():
        valid_loss, current_accuracy, current_norm_ED, preds, confidence_score, labels, infer_time, length_of_data = validation(
            model, criterion, val_loader, converter, opt)
    model.train()

    # training loss and validation loss
    print("Train loss: ", loss_avg.val(), " Valid loss: ", valid_loss)
    loss_avg.reset()

    # keep best accuracy model (on valid dataset)
    if current_accuracy > best_accuracy:
        best_accuracy = current_accuracy
        torch.save(model.state_dict(), 'pretrained_model/best_accuracy.pth')
        print('best_accuracy: ', best_accuracy)
    if current_norm_ED > best_norm_ED:
        best_norm_ED = current_norm_ED
        torch.save(model.state_dict(), 'pretrained_model/best_norm_ED.pth')
        print('best_norm_ED: ', best_norm_ED)
    
    elapsed_time = time.time() - start_time
    print('Time cost: ', elapsed_time)

print("DONE!")

