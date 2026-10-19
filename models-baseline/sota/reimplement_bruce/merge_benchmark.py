import os
import pandas as pd

csv_path = '/home/nvsinh1/brats_segmentation/teb-net-model/models_evaluate/reimplement_bruce/output/benchmark_table_3.csv'
md_path = '/home/nvsinh1/brats_segmentation/teb-net-model/models_evaluate/reimplement_bruce/output/benchmark_table_3.md'

if not os.path.exists(csv_path):
    print("CSV file not found!")
    sys.exit(1)

df = pd.read_csv(csv_path)

# Ensure TEBNet-v9.1 entry is present
if 'TEBNet-v9.1' not in df['Model'].values:
    tebnet_v9 = {
        'Model': 'TEBNet-v9.1',
        'Train_MAE': 2.709668,
        'Test_MAE': 3.259564,
        'Train_RMSE': 4.307370,
        'Test_RMSE': 5.240977,
        'Train_R2': 0.994747,
        'Test_R2': 0.992104,
        'Time_sec': 10722.63
    }
    df = pd.concat([df, pd.DataFrame([tebnet_v9])], ignore_index=True)

df_sorted = df.sort_values(by='Test_MAE').reset_index(drop=True)
df_sorted.to_csv(csv_path, index=False)

md_content = "# Complete Benchmark Results: Bruce et al. (2025) vs TEBNet\n\n"
md_content += "Target Property: Standard Enthalpy of Formation ($\\Delta H^\\circ_{f,298}$) in `kcal/mol`  \n"
md_content += "Dataset: WUDILY-CHO (6,074 aliphatic C, H, O species)  \n"
md_content += "Evaluation Protocol: 10-Fold Cross-Validation  \n\n"

md_content += "| Rank | Model (Feature-Method) | Train_MAE | Test_MAE | Train_RMSE | Test_RMSE | Train_R2 | Test_R2 | Training Time (s) |\n"
md_content += "| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"

for idx, row in df_sorted.iterrows():
    is_tebnet = ('TEBNet' in str(row['Model']))
    bold_prefix = "**" if is_tebnet else ""
    bold_suffix = "**" if is_tebnet else ""
    md_content += f"| {idx+1} | {bold_prefix}{row['Model']}{bold_suffix} | {row['Train_MAE']:.2f} | {bold_prefix}{row['Test_MAE']:.2f}{bold_suffix} | {row['Train_RMSE']:.2f} | {bold_prefix}{row['Test_RMSE']:.2f}{bold_suffix} | {row['Train_R2']:.4f} | {row['Test_R2']:.4f} | {row['Time_sec']:.2f} |\n"

with open(md_path, 'w') as f:
    f.write(md_content)

print(f"Successfully updated complete benchmark table with {len(df_sorted)} models!")
