"""
Train all models or a specific model.

Usage:
    python scripts/train_models.py                    # Train all models
    python scripts/train_models.py --model ridge      # Train only Ridge
    python scripts/train_models.py --model random_forest
    python scripts/train_models.py --model gradient_boosting
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
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


def run(model_name: str = None):
    """Run complete training pipeline."""
    
    logger.info("="*60)
    logger.info("STARTING TRAINING PIPELINE")
    logger.info("="*60)
    
    # Load configs
    data_config = load_data_config()
    model_config = load_model_config()
    feature_config = load_feature_config()
    
    # Load data
    loader = DataLoader(data_config)
    raw_data = loader.load_all()
    
    # Validate
    validator = DataValidator()
    if not validator.validate_all(raw_data):
        raise ValueError("Data validation failed!")
    
    # Preprocess
    preprocessor = DataPreprocessor()
    clean_data = preprocessor.preprocess_all(raw_data)
    
    btc = clean_data['prices']['bitcoin']
    
    # Build features
    price_builder = PriceFeatureBuilder(feature_config.get('price_features'))
    macro_builder = MacroFeatureBuilder(feature_config.get('macro_features'))
    onchain_builder = OnchainFeatureBuilder(feature_config.get('onchain_features'))
    target_builder = TargetBuilder(
        horizon_days=data_config['target']['horizon_days']
    )
    
    price_features = price_builder.build(btc)
    macro_features = macro_builder.build(clean_data.get('vix'), clean_data.get('hy_spreads'))
    onchain_features = onchain_builder.build(clean_data.get('mvrv'), clean_data.get('volume'))
    target = target_builder.build(btc)
    
    # Combine and split
    selector = FeatureSelector()
    all_features = selector.combine_features(price_features, macro_features, onchain_features)
    dataset = selector.build_dataset(all_features, target)
    
    train_ratio = data_config['date_ranges']['train_ratio']
    X_train, X_test, y_train, y_test = selector.temporal_split(dataset, train_ratio)
    
    # Train models
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
        
        # Evaluate
        eval_results = evaluator.evaluate_model(model, X_train, y_train, X_test, y_test)
        results[name] = eval_results
        
        # Save model
        model.save(f"results/models/{name}_model.pkl")
        
        logger.info(f"{name} - Test R²: {eval_results['test']['r2']:.4f}")
    
    # Compare models
    if len(results) > 1:
        comparison = evaluator.compare_models(results)
        logger.info("\nModel Comparison:")
        logger.info(f"\n{comparison.to_string()}")
    
    logger.info("\nTraining pipeline complete!")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train prediction models')
    parser.add_argument('--model', type=str, default=None,
                       choices=['ridge', 'random_forest', 'gradient_boosting'],
                       help='Model to train (default: all)')
    args = parser.parse_args()
    
    run(model_name=args.model)