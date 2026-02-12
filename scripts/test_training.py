# scripts/test_training.py
import sys
sys.path.append('.')

print("Script started...")

try:
    # ============================================================
    # IMPORTS
    # ============================================================
    print("Importing modules...")
    from src.data.loader import DataLoader
    from src.data.preprocessor import DataPreprocessor
    from src.features.price_features import PriceFeatureBuilder
    from src.features.macro_features import MacroFeatureBuilder
    from src.features.onchain_features import OnchainFeatureBuilder
    from src.features.target_builder import TargetBuilder
    from src.features.feature_selector import FeatureSelector
    from src.training.trainer import ModelTrainer
    from src.evaluation.metrics import ModelEvaluator
    from src.utils.config_loader import load_feature_config, load_data_config, load_model_config
    print("✓ All imports OK")

except Exception as e:
    print(f"✗ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================
# PHASES 1 & 2 (already verified - run quickly)
# ============================================================
try:
    print("\nRunning Phase 1 & 2 (data + features)...")

    data_config = load_data_config()
    feature_config = load_feature_config()
    model_config = load_model_config()

    # Load & preprocess
    loader = DataLoader()
    data = loader.load_all()
    preprocessor = DataPreprocessor()
    clean_data = preprocessor.preprocess_all(data)
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

    print(f"✓ Data ready - Train: {X_train.shape}, Test: {X_test.shape}")

except Exception as e:
    print(f"✗ Phase 1/2 failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================
# PHASE 3: TRAIN RIDGE
# ============================================================
try:
    print("\n" + "="*60)
    print("TRAINING RIDGE")
    print("="*60)

    trainer = ModelTrainer(config=model_config)
    evaluator = ModelEvaluator()

    ridge_model = trainer.train_ridge(X_train, y_train)
    ridge_results = evaluator.evaluate_model(
        ridge_model, X_train, y_train, X_test, y_test
    )

    print(f"\n✓ Ridge Results:")
    print(f"  Train R²:  {ridge_results['train']['r2']:.4f}")
    print(f"  Test R²:   {ridge_results['test']['r2']:.4f}")
    print(f"  Train RMSE: {ridge_results['train']['rmse']:.4f}%")
    print(f"  Test RMSE:  {ridge_results['test']['rmse']:.4f}%")
    print(f"  Overfitting gap: {ridge_results['overfitting_gap']:.4f}")

    # Save
    ridge_model.save("results/models/ridge_model.pkl")
    print(f"✓ Ridge model saved")

except Exception as e:
    print(f"✗ Ridge training failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================
# PHASE 3: TRAIN RANDOM FOREST
# ============================================================
try:
    print("\n" + "="*60)
    print("TRAINING RANDOM FOREST")
    print("="*60)

    rf_model = trainer.train_random_forest(X_train, y_train)
    rf_results = evaluator.evaluate_model(
        rf_model, X_train, y_train, X_test, y_test
    )

    print(f"\n✓ Random Forest Results:")
    print(f"  Train R²:  {rf_results['train']['r2']:.4f}")
    print(f"  Test R²:   {rf_results['test']['r2']:.4f}")
    print(f"  Train RMSE: {rf_results['train']['rmse']:.4f}%")
    print(f"  Test RMSE:  {rf_results['test']['rmse']:.4f}%")
    print(f"  Overfitting gap: {rf_results['overfitting_gap']:.4f}")

    # Save
    rf_model.save("results/models/rf_model.pkl")
    print(f"✓ Random Forest model saved")

except Exception as e:
    print(f"✗ Random Forest training failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================
# PHASE 3: TRAIN GRADIENT BOOSTING
# ============================================================
try:
    print("\n" + "="*60)
    print("TRAINING GRADIENT BOOSTING")
    print("="*60)

    gb_model = trainer.train_gradient_boosting(X_train, y_train)
    gb_results = evaluator.evaluate_model(
        gb_model, X_train, y_train, X_test, y_test
    )

    print(f"\n✓ Gradient Boosting Results:")
    print(f"  Train R²:  {gb_results['train']['r2']:.4f}")
    print(f"  Test R²:   {gb_results['test']['r2']:.4f}")
    print(f"  Train RMSE: {gb_results['train']['rmse']:.4f}%")
    print(f"  Test RMSE:  {gb_results['test']['rmse']:.4f}%")
    print(f"  Overfitting gap: {gb_results['overfitting_gap']:.4f}")

    # Save
    gb_model.save("results/models/gb_model.pkl")
    print(f"✓ Gradient Boosting model saved")

except Exception as e:
    print(f"✗ Gradient Boosting training failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================
# MODEL COMPARISON
# ============================================================
try:
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)

    all_results = {
        'ridge': ridge_results,
        'random_forest': rf_results,
        'gradient_boosting': gb_results
    }

    comparison = evaluator.compare_models(all_results)

    print(f"\n{comparison.to_string()}")

    # Save comparison
    comparison.to_csv("results/metrics/model_comparison.csv")
    print(f"\n✓ Comparison saved to results/metrics/model_comparison.csv")

except Exception as e:
    print(f"✗ Model comparison failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("✓ Phase 3 complete!")
print("="*60)