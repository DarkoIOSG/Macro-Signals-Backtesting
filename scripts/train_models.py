"""
Train all models or a specific model.

Usage:
    python scripts/train_models.py                     # Train all models
    python scripts/train_models.py --model ridge
    python scripts/train_models.py --model random_forest
    python scripts/train_models.py --model gradient_boosting
"""

import sys
import argparse
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.data.validator import DataValidator
from src.features.price_features import PriceFeatureBuilder
from src.features.macro_features import MacroFeatureBuilder
from src.features.onchain_features import OnchainFeatureBuilder
from src.features.target_builder import TargetBuilder
from src.features.feature_selector import FeatureSelector
from src.training.trainer import ModelTrainer
from src.evaluation.metrics import ModelEvaluator
from src.utils.config_loader import load_data_config, load_model_config, load_feature_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_dataset():
    """Run full data + feature pipeline and return train/test splits."""

    data_config = load_data_config()
    feature_config = load_feature_config()

    # Load & validate & preprocess
    loader = DataLoader(data_config)
    raw_data = loader.load_all()

    validator = DataValidator()
    if not validator.validate_all(raw_data):
        raise ValueError("Data validation failed!")

    preprocessor = DataPreprocessor()
    clean_data = preprocessor.preprocess_all(raw_data)
    btc = clean_data['prices']['bitcoin']

    # Build features
    price_features = PriceFeatureBuilder(
        feature_config.get('price_features')
    ).build(btc)

    macro_features = MacroFeatureBuilder(
        feature_config.get('macro_features')
    ).build(
        df_vix=clean_data.get('vix'),
        df_hy=clean_data.get('hy_spreads')
    )

    onchain_features = OnchainFeatureBuilder(
        feature_config.get('onchain_features')
    ).build(
        df_mvrv=clean_data.get('mvrv'),
        df_volume=clean_data.get('volume')
    )

    target = TargetBuilder(
        horizon_days=data_config['target']['horizon_days']
    ).build(btc)

    # Combine & split
    selector = FeatureSelector()
    all_features = selector.combine_features(
        price_features, macro_features, onchain_features
    )
    dataset = selector.build_dataset(all_features, target)
    X_train, X_test, y_train, y_test = selector.temporal_split(
        dataset, data_config['date_ranges']['train_ratio']
    )

    logger.info(f"Dataset ready - Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def run(model_name: str = None):
    """Run complete training pipeline."""

    logger.info("=" * 60)
    logger.info("STARTING TRAINING PIPELINE")
    logger.info("=" * 60)

    model_config = load_model_config()

    # Build dataset
    X_train, X_test, y_train, y_test = build_dataset()

    # Train
    trainer = ModelTrainer(config=model_config)
    evaluator = ModelEvaluator()

    models_to_train = (
        ['ridge', 'random_forest', 'gradient_boosting']
        if model_name is None else [model_name]
    )

    results = {}
    for name in models_to_train:
        logger.info(f"\nTraining {name}...")

        train_func = getattr(trainer, f"train_{name}")
        model = train_func(X_train, y_train)

        eval_results = evaluator.evaluate_model(
            model, X_train, y_train, X_test, y_test
        )
        results[name] = eval_results

        model.save(f"results/models/{name}_model.pkl")
        logger.info(f"{name} saved - Test R²: {eval_results['test']['r2']:.4f}")

    # Compare all models
    if len(results) > 1:
        comparison = evaluator.compare_models(results)
        logger.info("\nModel Comparison:")
        logger.info(f"\n{comparison.to_string()}")
        comparison.to_csv("results/metrics/model_comparison.csv")
        logger.info("Comparison saved to results/metrics/model_comparison.csv")
    else:
        # Single model - save its metrics too
        name = models_to_train[0]
        r = results[name]
        logger.info(f"\n{name} Results:")
        logger.info(f"  Train R²:        {r['train']['r2']:.4f}")
        logger.info(f"  Test R²:         {r['test']['r2']:.4f}")
        logger.info(f"  Train RMSE:      {r['train']['rmse']:.4f}%")
        logger.info(f"  Test RMSE:       {r['test']['rmse']:.4f}%")
        logger.info(f"  Overfitting gap: {r['overfitting_gap']:.4f}")

    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE!")
    logger.info("=" * 60)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Bitcoin return prediction models')
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        choices=['ridge', 'random_forest', 'gradient_boosting'],
        help='Model to train (default: all)'
    )
    args = parser.parse_args()
    run(model_name=args.model)