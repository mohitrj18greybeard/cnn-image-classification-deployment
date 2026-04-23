"""Training engine with validation, scheduling, early stopping, and metric tracking."""
import os, time, logging, torch, torch.nn as nn, torch.optim as optim
import numpy as np
import mlflow
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from tqdm import tqdm

log = logging.getLogger(__name__)

class Trainer:
    def __init__(self, config, device=None):
        self.cfg = config["training"]
        self.device = device or torch.device("cpu")

    def _build_optimizer(self, model):
        return optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                          lr=self.cfg["learning_rate"], weight_decay=self.cfg["weight_decay"])

    def _build_scheduler(self, optimizer):
        sc = self.cfg["scheduler"]
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=sc["T_max"], eta_min=sc["eta_min"])

    def train(self, model, train_loader, val_loader, save_path, name="model", epochs=None):
        model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = self._build_optimizer(model)
        scheduler = self._build_scheduler(optimizer)
        epochs = epochs or self.cfg["epochs"]
        best_acc, patience_counter = 0.0, 0
        es = self.cfg["early_stopping"]
        history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": [], "lr": []}

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        log.info(f"Training {name} on {self.device} for {epochs} epochs")
        
        mlflow.set_experiment("PlantGuard AI")
        with mlflow.start_run(run_name=name):
            # Log hyperparameters
            mlflow.log_params(self.cfg)
            mlflow.log_param("model_name", name)
            mlflow.log_param("device", str(self.device))

            t0 = time.time()
            for epoch in range(epochs):
                model.train()
                running_loss, correct, total = 0.0, 0, 0
                pbar = tqdm(train_loader, desc=f"[{name}] Epoch {epoch+1}/{epochs}", leave=False)
                for inputs, labels in pbar:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    optimizer.zero_grad()
                    loss = criterion(model(inputs), labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    running_loss += loss.item()
                    _, pred = model(inputs).max(1)
                    total += labels.size(0); correct += pred.eq(labels).sum().item()
                    pbar.set_postfix(loss=f"{running_loss/max(1,pbar.n):.4f}", acc=f"{100.*correct/total:.1f}%")

                train_loss = running_loss / len(train_loader)
                train_acc = correct / total
                val_loss, val_acc = self._validate(model, val_loader, criterion)
                lr = optimizer.param_groups[0]["lr"]
                scheduler.step()

                history["loss"].append(train_loss); history["accuracy"].append(train_acc)
                history["val_loss"].append(val_loss); history["val_accuracy"].append(val_acc)
                history["lr"].append(lr)
                
                # Log to MLflow
                mlflow.log_metric("train_loss", train_loss, step=epoch)
                mlflow.log_metric("train_acc", train_acc, step=epoch)
                mlflow.log_metric("val_loss", val_loss, step=epoch)
                mlflow.log_metric("val_acc", val_acc, step=epoch)
                mlflow.log_metric("lr", lr, step=epoch)

                log.info(f"Epoch {epoch+1}: loss={train_loss:.4f} acc={train_acc:.4f} "
                         f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} lr={lr:.6f}")

                if val_acc > best_acc + es["min_delta"]:
                    best_acc = val_acc; patience_counter = 0
                    torch.save(model.state_dict(), save_path)
                    log.info(f"  ✓ Saved best model ({best_acc:.4f})")
                else:
                    patience_counter += 1
                    if patience_counter >= es["patience"]:
                        log.info(f"  Early stopping at epoch {epoch+1}"); break

            # Log final best model
            mlflow.pytorch.log_model(model, "model")
            
        elapsed = time.time() - t0
        return {"history": history, "best_val_accuracy": best_acc,
                "training_time": elapsed, "epochs_trained": epoch + 1}

    def _validate(self, model, loader, criterion):
        model.eval()
        loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                out = model(x); loss += criterion(out, y).item()
                _, pred = out.max(1); total += y.size(0); correct += pred.eq(y).sum().item()
        return loss / len(loader), correct / total

    def evaluate(self, model, loader, class_names):
        model.to(self.device); model.eval()
        all_preds, all_labels, all_probs = [], [], []
        with torch.no_grad():
            for x, y in loader:
                out = model(x.to(self.device))
                probs = torch.softmax(out, 1)
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(out.argmax(1).cpu().numpy())
                all_labels.extend(y.numpy())
        y_true, y_pred = np.array(all_labels), np.array(all_preds)
        acc = accuracy_score(y_true, y_pred)
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        per_class = {class_names[i]: float(cm[i, i] / max(cm[i].sum(), 1)) for i in range(len(class_names))}
        return {"accuracy": acc, "precision": p, "recall": r, "f1_score": f1,
                "confusion_matrix": cm, "per_class_accuracy": per_class,
                "predictions": y_pred, "labels": y_true, "probabilities": np.array(all_probs)}
