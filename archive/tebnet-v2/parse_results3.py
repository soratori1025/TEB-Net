import json, glob
def check_mae(f):
    try:
        with open(f) as fp:
            data = json.load(fp)
            mae = data["test"]["all"]["H298_kcal"]["mae"]
            print(f"{mae:.2f} in {f}")
    except:
        pass
for f in glob.glob("../tebnet-v1/results/*/*/*/*/*/results.json"):
    check_mae(f)
for f in glob.glob("../tebnet-v1/results/*/*/*/*/results.json"):
    check_mae(f)
