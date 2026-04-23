"""
DeepVision CNN — Training Script (PyTorch Version)
=================================================
"""

import os
import argparse
import pickle
import logging
import torch
from src.utils.helpers import load_config, setup_logging, set_seed
from src.data.dataset import DatasetLoader
from src.models.custom_cnn import CustomCNN
from src.models.transfer_learning import TransferLearningModel
from src.training.trainer import ModelTrainer
from src.evaluation.evaluator import ModelEvaluator

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="both", choices=["custom", "transfer", "both"])
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    setup_logging("INFO")
    set_seed(42)
    config = load_config()
    if args.quick:
        config["training"]["epochs"] = 2
        logger.info("Quick mode enabled: 2 epochs.")

    loader = DatasetLoader(config)
    trainer = ModelTrainer(config)
    evaluator = ModelEvaluator(loader.get_class_names())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.quick:
        # Override loader to return small subsets
        def get_quick_loaders(for_transfer=False):
            train_ds, val_ds, test_ds = loader.load_data(for_transfer)
            from torch.utils.data import Subset
            train_ds = Subset(train_ds, range(200))
            val_ds = Subset(val_ds, range(100))
            test_ds = Subset(test_ds, range(100))
            from torch.utils.data import DataLoader
            return (DataLoader(train_ds, batch_size=32, shuffle=True),
                    DataLoader(val_ds, batch_size=32),
                    DataLoader(test_ds, batch_size=32))
        loader.get_loaders = get_quick_loaders

    results_all = {}
    histories_all = {}

    if args.model in ["custom", "both"]:
        model = CustomCNN(config)
        train_l, val_l, test_l = loader.get_loaders(for_transfer=False)
        res = trainer.train(model, train_l, val_l, config["paths"]["custom_model"], "Custom CNN")
        
        # Load best and evaluate
        model.load_state_dict(torch.load(config["paths"]["custom_model"]))
        eval_res = evaluator.evaluate_model(model, test_l, device)
        results_all["Custom CNN"] = eval_res
        histories_all["Custom CNN"] = res

    if args.model in ["transfer", "both"]:
        model = TransferLearningModel(config)
        train_l, val_l, test_l = loader.get_loaders(for_transfer=True)
        res = trainer.train(model, train_l, val_l, config["paths"]["transfer_model"], "Transfer Learning")
        
        # Load best and evaluate
        model.load_state_dict(torch.load(config["paths"]["transfer_model"]))
        eval_res = evaluator.evaluate_model(model, test_l, device)
        results_all["Transfer Learning"] = eval_res
        histories_all["Transfer Learning"] = res

    # Save results
    comparison = {"histories": histories_all, "results": results_all}
    with open(config["paths"]["comparison_results"], "wb") as f:
        pickle.dump(comparison, f)

if __name__ == "__main__":
    main()
