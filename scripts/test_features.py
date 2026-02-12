# scripts/test_features.py
import sys
sys.path.append('.')

print("Script started...")

try:
    print("Importing modules...")
    from src.data.loader import DataLoader
    from src.data.preprocessor import DataPreprocessor
    print("✓ Data imports OK")

    from src.features.price_features import PriceFeatureBuilder
    from src.features.macro_features import MacroFeatureBuilder
    from src.features.onchain_features import OnchainFeatureBuilder
    from src.features.target_builder import TargetBuilder
    from src.features.feature_selector import FeatureSelector
    print("✓ Feature imports OK")

    from src.utils.config_loader import load_feature_config, load_data_config
    print("✓ Config imports OK")

except Exception as e:
    print(f"✗ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================
# LOAD & PREPROCESS
# ============================================================
try:
    print("\nLoading data...")
    loader = DataLoader()
    data = loader.load_all()
    print(f"✓ Data loaded")

    preprocessor = DataPreprocessor()
    clean_data = preprocessor.preprocess_all(data)
    print(f"✓ Data preprocessed")

    btc = clean_data['prices']['bitcoin']
    print(f"✓ BTC series: {len(btc)} rows")

except Exception as e:
    print(f"✗ Data loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================
# PRICE FEATURES
# ============================================================
try:
    print("\nBuilding price features...")
    feature_config = load_feature_config()
    price_builder = PriceFeatureBuilder(feature_config.get('price_features'))
    price_features = price_builder.build(btc)
    print(f"✓ Price features: {price_features.shape}")

except Exception as e:
    print(f"✗ Price features failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================
# MACRO FEATURES
# ============================================================
try:
    print("\nBuilding macro features...")
    macro_builder = MacroFeatureBuilder(feature_config.get('macro_features'))
    macro_features = macro_builder.build(
        df_vix=clean_data.get('vix'),
        df_hy=clean_data.get('hy_spreads')
    )
    print(f"✓ Macro features: {macro_features.shape}")

except Exception as e:
    print(f"✗ Macro features failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================
# ON-CHAIN FEATURES
# ============================================================
try:
    print("\nBuilding on-chain features...")
    onchain_builder = OnchainFeatureBuilder(feature_config.get('onchain_features'))
    onchain_features = onchain_builder.build(
        df_mvrv=clean_data.get('mvrv'),
        df_volume=clean_data.get('volume')
    )
    print(f"✓ On-chain features: {onchain_features.shape}")

except Exception as e:
    print(f"✗ On-chain features failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================
# TARGET
# ============================================================
try:
    print("\nBuilding target variable...")
    data_config = load_data_config()
    target_builder = TargetBuilder(
        horizon_days=data_config['target']['horizon_days']
    )
    target = target_builder.build(btc)
    print(f"✓ Target: {target.notna().sum()} valid values")

except Exception as e:
    print(f"✗ Target building failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================
# COMBINE & SPLIT
# ============================================================
try:
    print("\nCombining features and splitting...")
    selector = FeatureSelector()

    all_features = selector.combine_features(
        price_features,
        macro_features,
        onchain_features
    )
    print(f"✓ Combined features: {all_features.shape}")

    dataset = selector.build_dataset(all_features, target)
    print(f"✓ Clean dataset: {dataset.shape}")
    print(f"  Date range: {dataset.index[0].date()} to {dataset.index[-1].date()}")

    train_ratio = data_config['date_ranges']['train_ratio']
    X_train, X_test, y_train, y_test = selector.temporal_split(dataset, train_ratio)

    print(f"✓ Train: {X_train.shape} ({X_train.index[0].date()} to {X_train.index[-1].date()})")
    print(f"✓ Test:  {X_test.shape} ({X_test.index[0].date()} to {X_test.index[-1].date()})")

except Exception as e:
    print(f"✗ Combining/splitting failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✓ Phase 2 complete!")