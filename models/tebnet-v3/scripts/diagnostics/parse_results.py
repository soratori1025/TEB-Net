import json, glob
def get_mae(pattern):
    maes = []
    for f in glob.glob(pattern):
        try:
            with open(f) as fp:
                data = json.load(fp)
                maes.append((f, data["test"]["all"]["H298_kcal"]["mae"]))
        except:
            pass
    return maes
print("TEB-Net:")
for f, m in get_mae("benchmark_results_deeptherm/thermo_none_chebyshev_3d_grouped_d5_global_w1-1-1_mon-h/random0/member0_seed*/results.json"):
    print(f, m)
print("No prior:")
for f, m in get_mae("benchmark_results_deeptherm/thermo_none_chebyshev_3d_grouped_d5_global_w1-0-0_mon-h/random0/member0_seed*/results.json"):
    print(f, m)
print("Prior only:")
for f, m in get_mae("benchmark_results_deeptherm/thermo_none_prior_v1_d5_global_w1-1-1_mon-h/random0/member0_seed*/results.json"):
    print(f, m)
