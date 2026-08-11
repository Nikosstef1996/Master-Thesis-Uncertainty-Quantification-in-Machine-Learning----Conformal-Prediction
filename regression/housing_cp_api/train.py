from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from app.model_service import train_all_models, save_model_registry


# 1. Load housing dataset
housing = fetch_california_housing(as_frame=True)

X = housing.data
y = housing.target

# 2. First split: train+calibration vs test
X_temp, X_test, y_temp, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 3. Second split: train vs calibration
X_train, X_calib, y_train, y_calib = train_test_split(
    X_temp,
    y_temp,
    test_size=0.25,
    random_state=42
)

# Final proportions:
# train = 60%
# calibration = 20%
# test = 20%

# 4. Train conformal models
results = train_all_models(
    X_train=X_train,
    y_train=y_train,
    X_calib=X_calib,
    y_calib=y_calib,
    X_test=X_test,
    y_test=y_test,
    alpha=0.1
)

# 5. Save trained models
save_model_registry(results)

# 6. Print results
for name, result in results.items():
    print(f"\n=== {name.upper()} ===")
    print(f"q_hat    : {result['q_hat']:.6f}")
    print(f"coverage : {result['coverage']:.6f}")
    print(f"avg width: {result['avg_width']:.6f}")
