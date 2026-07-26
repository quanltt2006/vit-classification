import torch
from tqdm import tqdm
from utils.metrics import calculate_accuracy

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    running_acc = 0.0

    pbar = tqdm(dataloader, desc = "Training")

    for images, labels in pbar: 
        images, labels = images.to(device), labels.to(device)


        outputs = model(images)

        loss = criterion(outputs,  labels)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        acc  = calculate_accuracy(outputs, labels)[0]

        running_loss += loss.item()
        running_acc += acc.item()

        pbar.set_postfix({'loss': loss.item(), 'acc': acc.item()})

    return running_loss/ len(dataloader), running_acc / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()

    running_loss = 0.0
    running_acc = 0.0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc = "Evaluating"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            acc = calculate_accuracy(outputs, labels)[0]

            running_loss += loss.item()
            running_acc += acc.item()

    return running_loss/len(dataloader), running_acc/len(dataloader)