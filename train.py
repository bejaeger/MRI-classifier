import logging

import torch
from torch.nn import CrossEntropyLoss,  Module
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader
from typing import Tuple

from dataset import CustomDataset
from data_utils import TorchDataset, pad_images
from model import SimpleCNN

logging.basicConfig(level=logging.INFO)
torch.manual_seed(55)

DATASET_FOLDER = "./data/train/"


def train(model: Module, train_loader: DataLoader, loss_fn: Module, optimizer: Optimizer, num_epochs: int) -> None:
    model.train()
    iteration_index = 0
    for epoch in range(num_epochs):
        for images, labels in train_loader:
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            iteration_index += 1
            logging.info(f'Iteration {iteration_index}, Loss: {loss.item()}')
        logging.info(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

def evaluate(model: Module, val_loader: DataLoader) -> Tuple[float, float]:
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in val_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

def main():
    # Hyperparameters
    num_epochs = 10
    batch_size = 4
    learning_rate = 0.0001

    logging.info("Loading data...")
    raw_dataset = CustomDataset(data_folder=DATASET_FOLDER, max_datapoints=16)

    logging.info("Preprocessing data...")
    images = pad_images(images=raw_dataset.image_data, target_depth=40)  # pad images to have depth 40 (found to be max depth)
    labels = raw_dataset.labels
    
    train_dataset = TorchDataset(images=images, labels=labels)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    logging.info("Loading model...")
    model = SimpleCNN()
    loss_fn = CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    logging.info("Start training...")
    train(model=model, train_loader=train_loader, loss_fn=loss_fn, optimizer=optimizer, num_epochs=num_epochs)
    accuracy = evaluate(model, train_loader)
    print(f'Train Accuracy: {accuracy}%')

if __name__ == "__main__":
    main()
