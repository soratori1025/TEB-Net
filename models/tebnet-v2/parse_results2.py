import json, glob

def get_mae(pattern):
    for f in glob.glob(pattern):
        try:
            with open(f) as fp:
                data = json.load(fp)
                print(f, data["test"]["all"]["H298_kcal"]["mae"])
        except Exception as e:
            pass

print("TEB-Net random0:")
get_mae("../tebnet-v1/benchmark_results_deeptherm/*/random0/member0_seed0/results.json")
print("TEB-Net complexity0:")
get_mae("../tebnet-v1/benchmark_results_deeptherm/*/complexity0/member0_seed0/results.json")
