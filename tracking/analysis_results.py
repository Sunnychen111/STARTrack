import _init_paths

# ============================================================
# Disable tikzplotlib before importing plot_results.py
# This avoids:
# ImportError: cannot import name 'common_texification'
# ============================================================
import sys
import types

tikz_stub = types.ModuleType("tikzplotlib")

def _dummy_save(*args, **kwargs):
    return None

def _dummy_clean_figure(*args, **kwargs):
    return None

def _dummy_get_tikz_code(*args, **kwargs):
    return ""

tikz_stub.save = _dummy_save
tikz_stub.clean_figure = _dummy_clean_figure
tikz_stub.get_tikz_code = _dummy_get_tikz_code
tikz_stub.Flavors = None

sys.modules["tikzplotlib"] = tikz_stub

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 8]

from lib.test.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from lib.test.evaluation import get_dataset, trackerlist


trackers = []
dataset_name = 'lasot'

trackers.extend(
    trackerlist(
        name='sutrack',
        parameter_name='startrack_5.9',
        dataset_name='lasot',
        run_ids=None,
        display_name='STARTrack_shadow_best'
    )
)

dataset = get_dataset(dataset_name)

print_results(
    trackers,
    dataset,
    dataset_name,
    merge_results=True,
    plot_types=('success', 'prec', 'norm_prec'),
    force_evaluation=True
)