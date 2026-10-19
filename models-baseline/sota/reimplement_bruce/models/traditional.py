from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

def get_traditional_model(model_name, random_state=42):
    """
    Factory function for conventional machine learning models from Bruce et al. (2025).
    """
    if model_name.endswith('-RF') or model_name == 'RF':
        return RandomForestRegressor(
            n_estimators=100,
            bootstrap=True,
            random_state=random_state,
            n_jobs=-1
        )
    elif model_name.endswith('-XGB') or model_name == 'XGB':
        return XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            random_state=random_state,
            n_jobs=-1
        )
    elif model_name.endswith('-SVR') or model_name == 'SVR':
        return SVR(
            kernel='rbf',
            C=10.0,
            epsilon=0.1
        )
    elif model_name.endswith('-DT') or model_name == 'DT':
        return DecisionTreeRegressor(
            random_state=random_state
        )
    elif model_name.endswith('-EXT') or model_name == 'EXT':
        return ExtraTreesRegressor(
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1
        )
    else:
        raise ValueError(f"Unknown traditional model name: {model_name}")
