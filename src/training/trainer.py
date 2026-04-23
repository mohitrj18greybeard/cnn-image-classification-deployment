"""
Model Training Pipeline (PyTorch Version)
========================================
Training loop, validation, and checkpointing for PyTorch models.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import logging
import time

logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epochs = config["training"]["epochs"]
        self.lr = config["training"]["learning_rate"]

    def train(self, model, train_loader, val_loader, save_path, model_name="model"):
        model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        best_acc = 0.0
        history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}

        logger.info(f"Starting training: {model_name} on {self.device}")

        for epoch in range(self.epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({"loss": f"{running_loss/len(train_loader):.4f}", "acc": f"{100.*correct/total:.2f}%"})

            train_loss = running_loss / len(train_loader)
            train_acc = correct / total
            
            # Validation
            val_loss, val_acc = self.validate(model, val_loader, criterion)
            scheduler.step(val_loss)
            
            history["loss"].append(train_loss)
            history["accuracy"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_acc)
            
            logger.info(f"Epoch {epoch+1}: Loss={train_loss:.4f}, Acc={train_acc:.4f}, ValLoss={val_loss:.4f}, ValAcc={val_acc:.4f}")

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), save_path)
                logger.info(f"Saved best model to {save_path}")

        return {"history": history, "best_val_accuracy": best_acc}

    def validate(self, model, loader, criterion):
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        return running_loss / len(loader), correct / total
