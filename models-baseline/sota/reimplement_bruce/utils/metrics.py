import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
def evaluate_predictions(y_true, y_pred):
    """
    Computes MAE, RMSE, and R2 score for true and predicted targets.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {
        'MAE': float(mae),
        'RMSE': float(rmse),
        'R2': float(r2)
    }
def summarize_cv_results(fold_results):
    """
    Summarizes metrics across k folds (mean and std).
    fold_results is a list of dicts containing train and test metrics.
    """
    summary = {}
    metric_keys = fold_results[0].keys()
    for key in metric_keys:
        vals = [res[key] for res in fold_results]
        summary[f'{key}_mean'] = float(np.mean(vals))
        summary[f'{key}_std'] = float(np.std(vals))
    return summary
