import json
def pr(f):
    try:
        with open(f) as fp:
            d = json.load(fp)
            print(f, d["test"]["all"]["H298_kcal"]["mae"])
    except: pass
pr("benchmark_results_deeptherm/thermo_none_chebyshev_3d_grouped_d5_global_w1-0-0_mon-h/random0/member0_seed0/results.json")
pr("benchmark_results_deeptherm/thermo_none_chebyshev_3d_grouped_d5_global_w1-0-0_mon-h/random0/member0_seed1/results.json")
pr("benchmark_results_deeptherm/thermo_none_chebyshev_3d_grouped_d5_global_w1-0-0_mon-h/random0/member0_seed2/results.json")
