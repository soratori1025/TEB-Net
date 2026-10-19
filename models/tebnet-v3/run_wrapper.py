import sys
import traceback
sys.argv = ['scripts/train.py', '--seed', '0', '--split_seed', '0', '--split_fractions', '0.1', '0.1', '0.8', '--csv', '/home/nvsinh1/brats_segmentation/teb-net-model/models-baseline/deeptherm_data.csv', '--s_prior', 'v2_exact', '--tag_suffix', '_frac0.1_splitH', '--loss_weights', '1', '0', '0', '--monitor', 'h', '--overwrite']
try:
    import scripts.train as t
    t.main()
except SystemExit as e:
    print(f"SYSTEM EXIT CAUGHT: {e.code}")
except Exception as e:
    print("EXCEPTION CAUGHT:")
    traceback.print_exc()
print("FINISHED WRAPPER")
