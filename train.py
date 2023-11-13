import os
import argparse
import logging
from typing import *

import torch
from torch.nn import CrossEntropyLoss,  Module
from torch.optim import AdamW
from torch.utils.data import DataLoader

from dataset import CustomDataset
from data_utils import TorchDataset, preprocess
from model import *

logging.basicConfig(level=logging.INFO)
torch.manual_seed(55)


def evaluate(model: Module, loader: DataLoader, loss_fn: Callable) -> Tuple[float, float]:
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        loss = 0
        for images, labels in loader:
            outputs = model(images)
            loss += loss_fn(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.shape[0]
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    loss = loss / len(loader)
    model.train()
    return accuracy, loss

def main(args: argparse.Namespace = None) -> None:
    # TODO: create config
    num_epochs = 10
    batch_size = 32
    learning_rate = 3e-4
    train_portion = 0.8

    dataset_folder = args.data_folder

    logging.info("Loading data...")
    raw_dataset = CustomDataset(data_folder=dataset_folder, max_datapoints=64)

    logging.info("Preprocessing data...")
    images = preprocess(images=raw_dataset.image_data)  # pad images to have depth 40 (found to be max depth)
    labels = raw_dataset.labels

    # TODO: Improve split to enusre equal amounts of positive labels in train and val
    train_images = images[:int(len(images) * train_portion)]
    train_labels = labels[:int(len(labels) * train_portion)]
    val_images = images[int(len(images) * train_portion):]
    val_labels = labels[int(len(labels) * train_portion):]

    logging.info(f"[train set] num positive/negative labels: {sum(train_labels)} / {len(train_labels) - sum(train_labels)}")
    logging.info(f"[val set] num positive/negative labels: {sum(val_labels)} / {len(val_labels) - sum(val_labels)}")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_dataset = TorchDataset(images=train_images, labels=train_labels)
    val_dataset = TorchDataset(images=val_images, labels=val_labels)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False)

    logging.info("Loading model...")
    model = AlexNet3D()
    loss_fn = CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    model.to(device)

    logging.info("Start training...")
    logging.info(model)
    model.train()
    iteration_index = 0
    for epoch in range(num_epochs):
        train_losses = []
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_losses.append(loss.item())

            iteration_index += 1
            logging.info(f'Iteration {iteration_index}, Loss: {loss.item()}')
        
        train_accuracy, _ = evaluate(model=model, loader=train_loader, loss_fn=loss_fn)
        val_accuracy, val_loss = evaluate(model=model, loader=val_loader, loss_fn=loss_fn)

        logging.info(
            f"[Epoch {epoch+1}/{num_epochs}] "
            f"Train loss: {torch.mean(torch.tensor(train_losses)):.3f}, Val loss: {val_loss:.3f}, "
            f"Train accuracy: {train_accuracy:.3f}, "
            f"Val accuracy: {val_accuracy:.3f}")

    output_path = "checkpoints/model.pth"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    logging.info(f"Saving model to `{output_path}`...")
    torch.save(model.state_dict(), "checkpoints/model.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data-folder', type=str, required=True)
    main(parser.parse_args())

