# scripts/test_loading.py
import sys
sys.path.append('.')

from src.data.loader import DataLoader
from src.data.validator import DataValidator
from src.data.preprocessor import DataPreprocessor

# ============================================================
# STEP 1: LOAD
# ============================================================
print("="*60)
print("STEP 1: LOADING DATA")
print("="*60)

loader = DataLoader()
data = loader.load_all()

print("Prices shape:", data['prices'].shape)
print("Prices columns:", data['prices'].columns.tolist())
print("Prices date range:", data['prices'].index[0], "to", data['prices'].index[-1])
print("\nVIX shape:", data['vix'].shape if data['vix'] is not None else "Not loaded")
print("HY shape:", data['hy_spreads'].shape if data['hy_spreads'] is not None else "Not loaded")
print("MVRV shape:", data['mvrv'].shape if data['mvrv'] is not None else "Not loaded")
print("Volume shape:", data['volume'].shape if data['volume'] is not None else "Not loaded")

# ============================================================
# STEP 2: VALIDATE
# ============================================================
print("\n" + "="*60)
print("STEP 2: VALIDATING DATA")
print("="*60)

validator = DataValidator()
is_valid = validator.validate_all(data)
print("Validation passed:", is_valid)

# ============================================================
# STEP 3: PREPROCESS
# ============================================================
print("\n" + "="*60)
print("STEP 3: PREPROCESSING DATA")
print("="*60)

preprocessor = DataPreprocessor()
clean_data = preprocessor.preprocess_all(data)

print("Prices shape:", clean_data['prices'].shape)
print("Prices date range:", 
      clean_data['prices'].index[0].date(), 
      "to", 
      clean_data['prices'].index[-1].date())

print("\n✓ Phase 1 complete!")