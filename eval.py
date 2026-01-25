import os
import pickle
import time
from itertools import combinations
import shutil

import numpy as np
from numpy import mean, std
import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments

# Import local modules
from SamplingAlgorithms import FeatureGuidedLogSampler, SequenceGuidedLogSampler, RandomLogSampler, \
    LongestTraceVariantLogSampler

# --- CONFIGURATION ---
DATA_CONFIG = {
    "Sepsis": {
        # Pointing directly to downloads folder
        "log_file": "downloads/Sepsis_Cases_-_Event_Log.xes",
        "model_file": "data/Sepsis_Cases.pnml",
        "index_file": "results/ICPM_2021/used_index_files/Sepsis_Cases_-_Event_Log.xes.index"
    },
    "BPI_2012": {
        # Pointing directly to downloads folder
        "log_file": "downloads/BPI_Challenge_2012.xes",
        "model_file": "data/BPI_Challenge_2012.pnml",
        "index_file": "results/ICPM_2021/used_index_files/BPI_Challenge_2012.xes.index"
    },
    "BPI_2018": {
        # IMPORTANT: This MUST point to data/ because we need the filtered version
        "log_file": "data/BPI_Challenge_2018_filtered.xes",
        "model_file": "data/BPI_Challenge_2018.pnml",
        "index_file": "results/ICPM_2021/used_index_files/BPI_Challenge_2018.xes.index"
    }
}

approaches = ["Random", "Longest", "Feature", "Sequence"]
samples_sizes = [100, 200, 300, 400, 500]
repetitions = 10
cached_alignments = False  # Set to False initially so it calculates them


def main():
    # Create output folders if they don't exist
    if not os.path.exists("results"): os.makedirs("results")
    if not os.path.exists("alignment_cache"): os.makedirs("alignment_cache")
    if not os.path.exists("knowledge_base_cache"): os.makedirs("knowledge_base_cache")

    for name, paths in DATA_CONFIG.items():
        print(f"\n--- Processing {name} ---")

        # Load inputs using the specific paths
        log, model, im, fm = load_inputs(paths["log_file"], paths["model_file"])

        # Run evaluations
        # 1. Partition stats
        eval_partitions(log, name, paths["index_file"])

        # 2. Quality (The main Figure 3 results)
        eval_quality(log, name, model, im, fm, paths["index_file"])

        # 3. Runtime
        eval_runtime(log, name, model, im, fm)


def load_inputs(log_path, model_path):
    print(f"Loading Log: {log_path}")
    log = xes_importer.apply(log_path)

    print(f"Loading Model: {model_path}")
    model, initial_marking, final_marking = pm4py.read_pnml(model_path)

    return log, model, initial_marking, final_marking


def eval_partitions(log, log_name, index_file):
    output_file = os.path.join("results", f"partitions_{log_name}.csv")
    with open(output_file, "w") as t:
        t.write("approach;total_size;partition;partition_size\n")

        for approach in approaches:
            if approach in ["Feature", "Sequence"]:
                print(f"Partitioning: {log_name} [{approach}]")

                if approach == "Feature":
                    sampler = FeatureGuidedLogSampler(log, index_file=index_file)
                elif approach == "Sequence":
                    sampler = SequenceGuidedLogSampler(log, batch_size=1, index_file=index_file)

                partitioning = sampler.partitioned_log
                for partition, traces in partitioning.items():
                    t.write(f"{approach};{len(partitioning)};{partition};{len(traces)}\n")


def eval_quality(log, log_name, model, initial_marking, final_marking, index_file):
    # Setup output files
    fitness_f = open(os.path.join("results", f"fitness_{log_name}.csv"), "w")
    fitness_f.write(
        "approach;sample_size;repetition;trace_variants;deviating_traces;total_deviations;num_deviating_activities;fitness;time_partitioning;time_sampling;time_alignments\n")

    # We skip detailed knowledge base tracking for speed, but keep the structure
    kb_f = open(os.path.join("results", f"knowledge_base_convergence_{log_name}.csv"), "w")
    kb_f.write("approach;sample_size;repetition;dist_to_baseline;first_positive_at;trace_idx;cor_change\n")

    corr_f = open(os.path.join("results", f"knowledge_base_correlations_{log_name}.csv"), "w")
    corr_f.write("approach;sample_size;repetition;feature;informative;correlation\n")

    act_f = open(os.path.join("results", f"activities_{log_name}.csv"), "w")
    act_f.write("approach;sample_size;avg_dev_activities;stddev;avg_pw_similarity;stddev\n")

    alignment_cache = {}

    for approach in approaches:
        for sample_size in samples_sizes:
            deviating_activities_list = []

            for i in range(repetitions):
                print(f"Quality Eval: {log_name} | {approach} | Size {sample_size} | Rep {i + 1}/{repetitions}")

                # Init Sampler
                if approach == "Random":
                    sampler = RandomLogSampler(use_cache=True)
                elif approach == "Longest":
                    sampler = LongestTraceVariantLogSampler(use_cache=True)
                elif approach == "Feature":
                    sampler = FeatureGuidedLogSampler(log, use_cache=True, index_file=index_file)
                elif approach == "Sequence":
                    sampler = SequenceGuidedLogSampler(log, batch_size=1, use_cache=True, index_file=index_file)

                sampler.alignment_cache = alignment_cache

                # Run Sampling
                try:
                    sample = sampler.construct_sample(log, model, initial_marking, final_marking, sample_size)

                    # Update global cache with new alignments found
                    alignment_cache.update(sampler.alignment_cache)
                except Exception as e:
                    print(f"Error during sampling: {e}")
                    continue

                # Stats
                trace_variants = set()
                for trace in sample.traces:
                    rep = ">>".join([e["concept:name"] for e in trace])
                    trace_variants.add(rep)

                fitness_f.write(f"{approach};{sample_size};{i};{len(trace_variants)};"
                                f"{sample.trace_deviations};{sample.total_deviations};"
                                f"{len(sample.activity_deviations)};{sample.fitness};"
                                f"{sample.times.get('partitioning', 0)};{sample.times.get('sampling', 0)};"
                                f"{sample.times.get('alignment', 0)}\n")
                fitness_f.flush()

                # Collect deviating activities for stats
                deviating_activities_list.append(set(sample.activity_deviations.keys()))

            # Activity stats (Jaccard)
            if deviating_activities_list:
                sizes = [len(s) for s in deviating_activities_list]
                similarities = []
                for s1, s2 in combinations(deviating_activities_list, 2):
                    similarities.append(jaccard_sim(s1, s2))

                if not similarities: similarities = [0]

                act_f.write(
                    f"{approach};{sample_size};{mean(sizes)};{std(sizes)};{mean(similarities)};{std(similarities)}\n")

    fitness_f.close()
    kb_f.close()
    corr_f.close()
    act_f.close()


def eval_runtime(log, log_name, model, initial_marking, final_marking, timeout=1800):
    # Reduced timeout to 30 mins for reproducibility
    output_path = os.path.join("results", f"alignment_runtime_{log_name}.csv")
    f = open(output_path, "w")
    f.write("sample_size;time\n")

    for sample_size in samples_sizes:
        print(f"Runtime Eval: {log_name} | Size {sample_size}")

        # FIX: Initialize first, then set cache
        sampler = RandomLogSampler(use_cache=True)
        sampler.alignment_cache = {}

        try:
            sample = sampler.construct_sample(log, model, initial_marking, final_marking, sample_size)
            f.write(f"{sample_size};{sample.times['alignment']}\n")
        except Exception as e:
            print(f"Error in runtime eval: {e}")
            f.write(f"{sample_size};ERROR\n")

    f.close()


def jaccard_sim(s, t):
    if len(s) == 0 and len(t) == 0: return 1.0
    return float(len(s.intersection(t)) / float(len(s.union(t))))


def construct_alignment_param(model):
    # Basic cost function (1 for move on log/model, 0 for sync)
    model_cost = {t: 1 for t in model.transitions if t.label is not None}
    sync_cost = {t: 0 for t in model.transitions if t.label is not None}
    return {
        alignments.Parameters.PARAM_MODEL_COST_FUNCTION: model_cost,
        alignments.Parameters.PARAM_SYNC_COST_FUNCTION: sync_cost
    }


class ConstantList:
    def __init__(self, value): self.value = value

    def __getitem__(self, item): return self.value


if __name__ == '__main__':
    main()