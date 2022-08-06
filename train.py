import torch
import torchvision
from model.cnn import Classifier
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, random_split


def train_cnn(n_epochs=10, batch_size=16):
    # set seeds for reproducability
    seed_val = 42
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)


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
        batch_size=batch_size,
        num_workers=4
    )

    validation_dataloader = DataLoader(
        val_dataset,
        sampler=SequentialSampler(val_dataset),
        batch_size=batch_size,
        num_workers=4
    )

    # load model
    model = Classifier(n_out=62)
    if torch.cuda.is_available():
        model.cuda()

    # load loss, optimizer and other training variables
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    # run training loop for n_epochs epochs
    for epoch in range(n_epochs):
        total_train_loss = 0.0
        running_train_loss = 0.0
        model.train()
        print(f"Running training...")
        for i, (images, labels) in enumerate(train_dataloader, 0):
            model.zero_grad()
            output = model(images)
            loss = loss_func(output, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            running_train_loss += loss.item()

            if i % 2000 == 0:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_train_loss / i:.3f}')
                running_train_loss = 0.0
        
        avg_train_loss = total_train_loss / len(train_dataloader)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("")

        print("Running Validation...")
        model.eval()
        total_eval_loss = 0.0
        running_eval_loss = 0.0
        total = 0
        correct = 0

        for i, (images, labels) in enumerate(validation_dataloader, 0):
            with torch.no_grad():
                output = model(images)
            loss = loss_func(output, labels)
            total_eval_loss += loss.item()
            running_eval_loss += loss.item()

            _, predicted = torch.max(output, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if i % 2000 == 0:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_eval_loss / i:.3f}')
                print(f'[{epoch + 1}, {i + 1:5d}] total accuracy: {100 * correct / total:.3f}')
                running_eval_loss = 0.0

        avg_eval_loss = total_eval_loss / len(validation_dataloader)

        print("")
        print("  Average validation loss: {0:.2f}".format(avg_eval_loss))
        print("  Average validation accuracy: {0:.2f}".format(100 * correct // total))
        print("")

    torch.save(model.state_dict(), "data/trained_models/test.pth")


if __name__ == '__main__':
    train_cnn()