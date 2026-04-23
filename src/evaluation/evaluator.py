"""
Model Evaluation Module (PyTorch Version)
========================================
Comprehensive evaluation using PyTorch.
"""

import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logger = logging.getLogger(__name__)


class ModelEvaluator:
    def __init__(self, class_names):
        self.class_names = class_names
        plt.style.use('dark_background')

    def evaluate_model(self, model, loader, device):
        model.to(device)
        model.eval()
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        accuracy = accuracy_score(all_labels, all_preds)
        cm = confusion_matrix(all_labels, all_preds)
        report = classification_report(all_labels, all_preds, target_names=self.class_names, output_dict=True)

        return {
            "accuracy": accuracy,
            "confusion_matrix": cm,
            "classification_report": report,
            "prediction_probabilities": all_probs,
            "predictions": all_preds,
            "labels": all_labels,
            "per_class_accuracy": {name: cm[i,i]/cm[i,:].sum() for i, name in enumerate(self.class_names)}
        }

    # Plotting methods remain largely similar, just need to ensure inputs are correct
    def plot_confusion_matrix(self, cm, title="Confusion Matrix"):
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=self.class_names, yticklabels=self.class_names, cmap='Blues')
        plt.title(title)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        return fig

    def plot_per_class_accuracy(self, per_class_acc, title="Per-Class Accuracy"):
        fig, ax = plt.subplots(figsize=(12, 6))
        names = list(per_class_acc.keys())
        accs = list(per_class_acc.values())
        ax.bar(names, accs, color='skyblue')
        plt.title(title)
        plt.xticks(rotation=45)
        return fig

    def plot_training_history(self, history, model_name="Model"):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        ax1.plot(history['loss'], label='Train Loss')
        ax1.plot(history['val_loss'], label='Val Loss')
        ax1.set_title(f'{model_name} Loss')
        ax1.legend()
        
        ax2.plot(history['accuracy'], label='Train Acc')
        ax2.plot(history['val_accuracy'], label='Val Acc')
        ax2.set_title(f'{model_name} Accuracy')
        ax2.legend()
        return fig

    def plot_confidence_distribution(self, probs, labels, preds):
        fig, ax = plt.subplots(figsize=(10, 6))
        correct_probs = probs[range(len(labels)), labels][preds == labels]
        incorrect_probs = probs[range(len(labels)), labels][preds != labels]
        
        ax.hist(correct_probs, bins=50, alpha=0.5, label='Correct')
        ax.hist(incorrect_probs, bins=50, alpha=0.5, label='Incorrect')
        ax.set_title('Confidence Distribution')
        ax.legend()
        return fig
