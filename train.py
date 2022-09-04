from os import device_encoding
import json

import torch
import torchvision
from tensorboardX import SummaryWriter
from datetime import datetime
from easydict import EasyDict
from model.cnn import Classifier
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, random_split

from conf.config import CONF


def train_cnn():
    # set seeds for reproducability
    torch.manual_seed(CONF.TRAIN.MANUAL_SEED)
    torch.cuda.manual_seed_all(CONF.TRAIN.MANUAL_SEED)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.RandomRotation(degrees=(90, 90)),
        torchvision.transforms.RandomVerticalFlip(1),
        torchvision.transforms.ToTensor(),
    ])
    # load training and validation dataset
    emnist = torchvision.datasets.EMNIST("data", split="byclass", download=True, transform=transforms, train=True, )

    train_size = int(0.9 * len(emnist))
    val_size = len(emnist) - train_size

    train_dataset, val_dataset = random_split(
        emnist, [train_size, val_size])
    train_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=CONF.TRAIN.BATCH_SIZE,
        num_workers=CONF.TRAIN.NUM_WORKERS
    )

    validation_dataloader = DataLoader(
        val_dataset,
        sampler=SequentialSampler(val_dataset),
        batch_size=CONF.VAL.BATCH_SIZE,
        num_workers=CONF.VAL.NUM_WORKERS
    )

    # load model
    model = Classifier(MODEL_CONF = CONF.MODEL)

    #CUDA stuff
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    #logging
    writer = SummaryWriter("logs/{}".format(timestamp))
    with open("logs/{}/config.json".format(timestamp), 'w') as fp:
        json.dump(CONF, fp)

    # load loss, optimizer and other training variables
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    # run training loop for n_epochs epochs
    for epoch in range(CONF.TRAIN.EPOCHS):
        total_train_loss = 0.0
        running_train_loss = 0.0
        model.train()
        print(f"Running training...")
        for i, (images, labels) in enumerate(train_dataloader, 0):
            images = images.to(device)
            labels = labels.to(device)
            model.zero_grad()
            output = model(images)
            loss = loss_func(output, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            running_train_loss += loss.item()

            if i % 2000 == 0:    # print every 2000 mini-batches
                writer.add_scalar("training loss", running_train_loss/(i+1), i)
                running_train_loss = 0.0
        
        avg_train_loss = total_train_loss / len(train_dataloader)
        writer.add_scalar("Avg training loss over epoch", avg_train_loss, epoch)

        model.eval()
        total_eval_loss = 0.0
        running_eval_loss = 0.0
        total = 0
        correct = 0

        for i, (images, labels) in enumerate(validation_dataloader, 0):
            images = images.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                output = model(images)
            loss = loss_func(output, labels)
            total_eval_loss += loss.item()
            running_eval_loss += loss.item()

            _, predicted = torch.max(output, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if i % 2000 == 0:    # print every 2000 mini-batches
                #print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_eval_loss / i:.3f}')
                writer.add_scalar("val loss", running_eval_loss/(i+1), i)
                #print(f'[{epoch + 1}, {i + 1:5d}] total accuracy: {100 * correct / total:.3f}')
                writer.add_scalar("val accuracy", 100*correct/total, i)
                running_eval_loss = 0.0

        avg_eval_loss = total_eval_loss / len(validation_dataloader)

        writer.add_scalar("Average validation loss", avg_eval_loss, epoch)
        writer.add_scalar("Average validation accuracy", 100 * correct / total, epoch)

    torch.save(model.state_dict(), "logs/{}/test.pth".format(timestamp))


if __name__ == '__main__':
    train_cnn()