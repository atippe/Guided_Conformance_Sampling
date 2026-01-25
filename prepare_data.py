import os
from pathlib import Path

# Legacy PM4Py v2.2.x imports
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.objects.petri_net.exporter import exporter as pnml_exporter
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
from pm4py.objects.log.obj import EventLog

# Configuration mapping
CONFIG = [
    {
        "input": "downloads/BPI_Challenge_2012.xes",
        "output_model": "data/BPI_Challenge_2012.pnml",
        "filter": False
    },
    {
        "input": "downloads/Sepsis_Cases_-_Event_Log.xes",
        "output_model": "data/Sepsis_Cases.pnml",
        "filter": False
    },
    {
        "input": "downloads/BPI_Challenge_2018.xes",
        "output_model": "data/BPI_Challenge_2018.pnml",
        "filter": True
    }
]


def prepare_log(cfg):
    input_path = Path(cfg["input"])
    output_model = Path(cfg["output_model"])

    # Ensure data dir exists
    output_model.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"SKIP: Input {input_path} not found. Did you run download_data.py?")
        return

    print(f"Processing {input_path}...")
    log = xes_importer.apply(str(input_path))

    # BPI 2018 specific filtering (Top 5% longest traces removed)
    if cfg["filter"]:
        print("  -> Filtering top 5% longest traces (BPI 2018)...")
        # Sort by length
        traces = sorted([t for t in log], key=lambda x: len(x))
        cutoff_idx = int(len(traces) * 0.95)
        cutoff_len = len(traces[cutoff_idx])

        # Keep only traces shorter than the 95th percentile
        filtered_list = [t for t in log if len(t) < cutoff_len]
        log = EventLog(filtered_list, attributes=log.attributes, classifiers=log.classifiers, extensions=log.extensions)

        # Save this filtered log because eval.py needs to sample from the *clean* version
        filtered_log_path = output_model.parent / (input_path.stem + "_filtered.xes")
        print(f"  -> Saving filtered log to {filtered_log_path}")
        xes_exporter.apply(log, str(filtered_log_path))

    # Discovery
    print(f"  -> Discovering Process Model (Inductive Miner IMf, 0.2 noise)...")
    net, im, fm = inductive_miner.apply(
        log,
        variant=inductive_miner.Variants.IMf,
        parameters={inductive_miner.Variants.IMf.value.Parameters.NOISE_THRESHOLD: 0.2}
    )

    print(f"  -> Exporting model to {output_model}")
    pnml_exporter.apply(net, im, str(output_model))


if __name__ == "__main__":
    for config in CONFIG:
        prepare_log(config)