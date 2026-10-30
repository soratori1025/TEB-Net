import os
import pandas as pd
from run_single_model import run_single_model_cv
CDS_MODELS = [
    'CDS-RF',
    'CDS-XGB',
    'CDS-ANN-KR',
    'CDS-DT',
    'CDS-EXT'
]
def main():
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    csv_path = os.path.join(output_dir, 'benchmark_table_3.csv')
    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
    else:
        print("Existing benchmark_table_3.csv not found. Please run the full benchmark first.")
        return
    print("\n=========================================================================")
    print("RE-RUNNING CDS MODELS WITH UFF & 3D DESCRIPTORS (RANDOM SPLIT)")
    print("=========================================================================\n")
    new_results = []
    for model_name in CDS_MODELS:
        try:
            epochs = 100 if ('GNF' in model_name or 'ANN' in model_name) else 0
            res = run_single_model_cv(model_name, n_splits=10, seed=42, epochs=epochs)
            new_results.append(res)
        except Exception as e:
            print(f"Error running model {model_name}: {e}")
    df_new = pd.DataFrame(new_results)
    for _, row in df_new.iterrows():
        model_idx = df_existing[df_existing['Model'] == row['Model']].index
        if len(model_idx) > 0:
            for col in df_new.columns:
                df_existing.loc[model_idx, col] = row[col]
        else:
            df_existing = pd.concat([df_existing, pd.DataFrame([row])], ignore_index=True)
    df_sorted = df_existing.sort_values(by='Test_MAE').reset_index(drop=True)
    df_sorted.to_csv(csv_path, index=False)
    print("\n=========================================================================")
    print("UPDATED BENCHMARK RESULTS (WUDILY-CHO DATASET - 10-FOLD CV)")
    print("=========================================================================\n")
    print(df_sorted.to_string(index=False))
    print(f"\nUpdated CSV results saved to {csv_path}")
if __name__ == '__main__':
    main()
