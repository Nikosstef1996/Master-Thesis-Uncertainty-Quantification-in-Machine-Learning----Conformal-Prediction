from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from app.model_service import train_all_models, save_model_registry


# 1. Load Iris dataset
iris = load_iris()

X = iris.data
y = iris.target
target_names = iris.target_names

# 2. Split into train+calibration and test
X_temp, X_test, y_temp, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3. Split train+calibration into train and calibration
X_train, X_calib, y_train, y_calib = train_test_split(
    X_temp,
    y_temp,
    test_size=0.25,
    random_state=42,
    stratify=y_temp
)

# Final proportions:
# train = 60%
# calibration = 20%
# test = 20%

# 4. Train conformal classifiers
results = train_all_models(
    X_train=X_train,
    y_train=y_train,
    X_calib=X_calib,
    y_calib=y_calib,
    X_test=X_test,
    y_test=y_test,
    target_names=target_names,
    alpha=0.1
)

# 5. Save trained models
save_model_registry(results)

# 6. Print evaluation
print("\n========== FINAL CLASSIFICATION EVALUATION ==========")

for name, result in results.items():
    print(f"\n--- Score function: {name} ---")
    print(f"q_hat: {result['q_hat']:.4f}")
    print(f"Coverage: {result['coverage']:.4f}")
    print(f"Average set size: {result['avg_set_size']:.4f}")