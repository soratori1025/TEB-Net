import os
import time
import pandas as pd
import numpy as np
from dataset import WudilyCHODataset
from run_single_model import run_single_model_cv
BENCHMARK_MODELS = [
    'CDS-RF',
    'MRD-RF',
    'ECFP-RF',
    'CM-RF',
    'SOAP-RF',
    'CDS-XGB',
    'MRD-XGB',
    'CDS-ANN-KR',
    'GNF-MPNN',
    'GNF-GCN',
    'GNF-GAT',
    'CDS-DT',
    'CDS-EXT',
    'CM-SVR',
    'ECFP-SVR',
    'MRD-SVR'
]
def main():
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, 'benchmark_table_3.csv')
    md_path = os.path.join(output_dir, 'benchmark_table_3.md')
    print("\n=========================================================================")
    print("STARTING FULL 10-FOLD CV BENCHMARK (BRUCE ET AL. 2025 RE-IMPLEMENTATION)")
    print("=========================================================================\n")
    results = []
    for model_name in BENCHMARK_MODELS:
        try:
            epochs = 100 if ('GNF' in model_name or 'ANN' in model_name) else 0
            res = run_single_model_cv(model_name, n_splits=10, seed=42, epochs=epochs)
            results.append(res)
            df_res = pd.DataFrame(results)
            df_res.to_csv(csv_path, index=False)
        except Exception as e:
            print(f"Error running model {model_name}: {e}")
    df = pd.DataFrame(results)
    df_sorted = df.sort_values(by='Test_MAE').reset_index(drop=True)
    print("\n=========================================================================")
    print("FINAL RE-IMPLEMENTED BENCHMARK RESULTS (WUDILY-CHO DATASET - 10-FOLD CV)")
    print("=========================================================================\n")
    print(df_sorted.to_string(index=False))
    md_content = "# Re-implemented Benchmark Results: Bruce et al. (2025)\n\n"
    md_content += f"Target Property: Standard Enthalpy of Formation (\\(\\Delta H^\\circ_{{f,298}}\\)) in `kcal/mol`\n"
    md_content += f"Dataset: WUDILY-CHO (6,074 aliphatic C, H, O species)\n"
    md_content += f"Evaluation Protocol: 10-Fold Cross-Validation\n\n"
    md_content += "| Model (Feature-Method) | Train_MAE | Test_MAE | Train_RMSE | Test_RMSE | Train_R2 | Test_R2 | Training Time (s) |\n"
    md_content += "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"
    for idx, row in df_sorted.iterrows():
        is_best = (idx == 0)
        bold_prefix = "**" if is_best else ""
        bold_suffix = "**" if is_best else ""
        md_content += f"| {bold_prefix}{row['Model']}{bold_suffix} | {row['Train_MAE']:.2f} | {bold_prefix}{row['Test_MAE']:.2f}{bold_suffix} | {row['Train_RMSE']:.2f} | {bold_prefix}{row['Test_RMSE']:.2f}{bold_suffix} | {row['Train_R2']:.4f} | {row['Test_R2']:.4f} | {row['Time_sec']:.2f} |\n"
    with open(md_path, 'w') as f:
        f.write(md_content)
    print(f"\nSaved CSV results to {csv_path}")
    print(f"Saved Markdown table to {md_path}")
if __name__ == '__main__':
    main()
