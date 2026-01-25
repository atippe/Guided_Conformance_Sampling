import itertools
import os
import statistics

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def main():
    #inputs = ["Sepsis_Cases_-_Event_Log.xes", "BPI_Challenge_2012.xes", "BPI_Challenge_2018.xes"]
    inputs = ["Sepsis"]
    base_dir = "results"
    sns.set_theme(style="ticks")

    for input_name in inputs:
        df = pd.read_csv(os.path.join(base_dir, "fitness_" + input_name + ".csv"), delimiter=";")
        plot_no_deviating_traces(df, input_name)
        # plot_impact_p_on_no_deviating_traces(df, input_name)
        # plot_constructed_alignments(df, input_name)
        # plot_fitness(df, input_name)
        # plot_preprocessing_sampling_runtimes(df, input_name)

    # for input_name in inputs:
    #   df = pd.read_csv(os.path.join(base_dir, "deviation_distribution_" + input_name + ".csv"), delimiter=";")
    #   plot_deviation_distributions(df, input_name)

    # for input_name in inputs:
    #   df = pd.read_csv(os.path.join(base_dir, "activities_" + input_name + ".csv"), delimiter=";")
    #   plot_deviating_activitiy_sets(df, input_name)
    #   plot_avg_set_similarity(df, input_name)

    for input_name in inputs:
        df = pd.read_csv(os.path.join(base_dir, "knowledge_base_convergence_" + input_name + ".csv"), delimiter=";")
        plot_knowledge_base_convergence(df, input_name)
        # plot_distance_to_baseline(df, input_name)

    # for input_name in inputs:
    #   df = pd.read_csv(os.path.join(base_dir, "knowledge_base_correlations_" + input_name + ".csv"), delimiter=";")
    #   plot_knowledge_base_correlations(df, input_name)

    # for input_name in inputs:
    #   df = pd.read_csv(os.path.join(base_dir, "partitions_" + input_name + ".csv"), delimiter=";")
    #   plot_partition_pruning(df, input_name)
    #   plot_partition_sizes(df, input_name)


def plot_no_deviating_traces(df, input_name):
    print("> Deviating Traces")
    # plt.figure(figsize=(6, 5))
    plt.ylim((-5, 510))

    g = sns.boxplot(x="sample_size", y="deviating_traces", hue="approach", data=df)
    plt.legend(loc='upper left', fontsize=18)

    g.legend_.texts[0].set_text('Random')
    g.legend_.texts[1].set_text('Longest')
    g.legend_.texts[2].set_text('Feature')
    g.legend_.texts[3].set_text('Behavioural')

    plt.xlabel("Sample size", fontsize=20)
    plt.ylabel("No of deviating traces", fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=18)

    plt.axvline(0.5, ls="--", c="lightgrey")
    plt.axvline(1.5, ls="--", c="lightgrey")
    plt.axvline(2.5, ls="--", c="lightgrey")
    plt.axvline(3.5, ls="--", c="lightgrey")
    plt.tight_layout()

    plt.savefig(os.path.join("results", "figures", "deviating_traces_" + input_name + ".pdf"), format="pdf")
    plt.clf()


def plot_knowledge_base_convergence(df, input_name):
    print("> Knowledge base convergence")
    subset_df = df.loc[df["sample_size"] == 500]

    plt.figure(figsize=(6, 3))
    g = sns.lineplot(x="trace_idx", y="cor_change", hue="approach", data=subset_df, ci=None)

    g.legend(loc='upper right', fontsize=18)
    g.legend_.texts[0].set_text('Feature')
    g.legend_.texts[1].set_text('Behavioural')

    plt.xlabel("No of sampled traces", fontsize=20)
    plt.ylabel("Abs. change", fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=18)

    # plt.ylim((-1,20))
    plt.xlim((-2, 505))
    # plt.axvline(statistics.mean(subset_df), ls="--", c="lightgrey")
    plt.tight_layout()

    plt.savefig(os.path.join("results", "figures", "knowledge_base_convergence_" + input_name + ".pdf"), format="pdf")
    plt.clf()

# ADDITIONAL UNUSED PLOTTING FUNCTIONS
# def plot_impact_p_on_no_deviating_traces(df, input_name):
#     """
#     Plots, for different index types, the number of buckets after prunign with different p
#     """
#     print("> Pruning Parameter Quality Impact")
#     subset_df = df.loc[df["sample_size"] == 500]
#     sns.boxplot(x="p", y="deviating_traces", hue="approach", data=subset_df)
#     plt.savefig(os.path.join("results", "figures", "pruning_impact_on_deviating_traces_" + input_name + ".pdf"), format="pdf")
#     plt.clf()
#
#
# def plot_constructed_alignments(df, input_name):
#     print("> Constructed alignments")
#     subset_df = df.loc[df["p"] == 5]
#     sns.boxplot(x="sample_size", y="trace_variants", hue="approach", data=subset_df)
#     plt.savefig(os.path.join("results", "figures", "constructed_alignments_" + input_name + ".pdf"), format="pdf")
#     plt.clf()
#
#
# def plot_fitness(df, input_name):
#     print("> Fitness")
#     sns.boxplot(x="sample_size", y="fitness", hue="approach", data=df)
#     plt.savefig(os.path.join("results", "figures", "fitness_" + input_name + ".pdf"), format="pdf")
#     plt.clf()
#
#
# def plot_deviation_distributions(df, input_name):
#     print("> Deviation Distribution")
#     activities = df["activity"].unique()
#
#     subset_df = df.loc[df["sample_size"] == 500]
#     subset_df = subset_df.loc[subset_df["activity"].isin(activities[:5])]
#
#     g = sns.boxplot(x="activity", y="prob", hue="approach", data=subset_df)
#
#     true_values = []
#     if input_name == "Sepsis_Cases_-_Event_Log.xes":
#         true_values = [0.191051136363636,
#                        0.169744318181818,
#                        0.137784090909091,
#                        0.112926136363636,
#                        0.082386363636364,
#                        0.090198863636364,
#                        0.017045454545455,
#                        0.039772727272727,
#                        0.057528409090909,
#                        0.0546875,
#                        0.017755681818182,
#                        0.022017045454546,
#                        0.004261363636364,
#                        0.002840909090909
#                        ]
#
#     if input_name == "BPI_Challenge_2012.xes":
#         true_values = [
#             0.282152057113417,
#             0.241745786413483,
#             0.082152057113417,
#             0.041318907779495,
#             0.037948038566277,
#             0.028218149701921,
#             0.028218149701921,
#             0.028218149701921,
#             0.015956428939427,
#             0.025053359829249,
#             0.019945536174284,
#             0.02757047177449,
#             0.027526311915802,
#             0.02757047177449,
#             0.02757047177449,
#             0.011805402222713,
#             0.015647309928608,
#             0.014410833885332,
#             0.003429749024803,
#             0.013365717229705,
#             0.000176639434754
#         ]
#
#     if input_name == "BPI_Challenge_2018.xes":
#         true_values = [
#             0.31188798900976,
#             0.18093161451095,
#             0.088496988865993,
#             0.036421143432214,
#             0.040965799268382,
#             0.003178862957461,
#             0.029344579160077,
#             0.04074216066836,
#             0.037683104103768,
#             0.039671890225396,
#             0.032363700260379,
#             0.022180156866504,
#             0.018562003801856,
#             0.017843165444642,
#             0.017851152537499,
#             0.006237919522052,
#             0.006693183814955,
#             0.007212344850721,
#             0.002428076228814,
#             0.005087778150509,
#             0.008985479465184,
#             0.001853005543042,
#             0.004249133400425,
#             0.004760307343333,
#             0.008705931215156,
#             0.007731505886487,
#             0.000567083592914,
#             0.006469545214933,
#             0.001541508921583,
#             0.003658088528937,
#             0.001230012300123,
#             0.002060669957349,
#             0.000367406271465,
#             0.000127793485727,
#             0.000375393364323,
#             0.000782735100078,
#             0.000239612785738,
#             0.000511173942908
#
#         ]
#     for i in range(5):
#         plt.hlines(true_values[i], i - .49, i + .49, colors="c")
#         # plt.plot([i-.5,i+.5], [true_values[i], true_values[i]], 'k-', lw=2)
#     # plt.margins(x=.1)
#     g.legend(loc='upper right', fontsize=18)
#     g.legend_.texts[0].set_text('Random')
#     g.legend_.texts[1].set_text('Longest')
#     g.legend_.texts[2].set_text('Feature')
#     g.legend_.texts[3].set_text('Behavioural')
#
#     plt.xlabel("Activity", fontsize=20)
#     plt.ylabel("Deviation frequency", fontsize=20)
#     plt.tick_params(axis='both', which='major', labelsize=18)
#     plt.xticks([0, 1, 2, 3, 4], [1, 2, 3, 4, 5])
#
#     plt.axvline(0.5, ls="--", c="lightgrey")
#     plt.axvline(1.5, ls="--", c="lightgrey")
#     plt.axvline(2.5, ls="--", c="lightgrey")
#     plt.axvline(3.5, ls="--", c="lightgrey")
#     plt.xlim(-.51, 4.51)
#     plt.ylim(-.1, .41)
#     plt.tight_layout()
#
#     plt.savefig(os.path.join("results", "figures",  "deviation_distribution_" + input_name + ".pdf"), format="pdf")
#     plt.clf()
#
#     activities = df["activity"].unique()
#     approaches = df["approach"].unique()
#     print(approaches)
#     print(activities)
#
#     for i in approaches:
#         subset_df = df.loc[df["approach"] == i]
#         subset_df = subset_df.loc[subset_df["sample_size"] == 500]
#         for j in activities:
#             activity_df = subset_df.loc[subset_df["activity"] == j]
#             # print(i,j,activity_df)
#             values = list(activity_df["prob"])
#
#             print(i, j, statistics.mean(values) if len(values) > 0 else "--",
#                   statistics.stdev(values) if len(values) > 1 else "--")
#         # groups = subset_df.groupby(subset_df.activity)
#
#
# def plot_deviating_activitiy_sets(df, input_name):
#     print("> Average deviating activitiy set sizes")
#     subset_df = df.loc[df["p"] == 5]
#     sns.lineplot(x="sample_size", y="avg_dev_activities", hue="approach", data=subset_df)
#     plt.savefig(os.path.join("results", "figures", "avg_dev_activities_" + input_name + ".pdf"), format="pdf")
#     plt.clf()
#
#
# def plot_avg_set_similarity(df, input_name):
#     print("> Average deviating activity set similarity")
#     subset_df = df.loc[df["p"] == 5]
#     sns.lineplot(x="sample_size", y="avg_pw_similarity", hue="approach", data=subset_df)
#     plt.savefig(os.path.join("results", "figures", "avg_pw_similarity_" + input_name + ".pdf"), format="pdf")
#     plt.clf()
#
#
# def plot_distance_to_baseline(df, input_name):
#     print("> Mean distance of knowledge base")
#     subset_df = df.loc[df["p"] == 5]
#     sns.boxplot(x="sample_size", y="dist_to_baseline", hue="approach", data=subset_df)
#     plt.savefig(os.path.join("results", "figures", "knowledge_base_distance_to_baseline_" + input_name + ".pdf"), format="pdf")
#     plt.clf()
#
#
# def plot_knowledge_base_correlations(df, input_name):
#     print("> Knowledge base correlations")
#     subset_df = df.loc[df["p"] == 5]
#     subset_df = subset_df.loc[subset_df["sample_size"] == 500]
#     subset_df = subset_df.loc[subset_df["informative"] == True]
#     sns.lineplot(x="feature", y="correlation", hue="approach", data=subset_df)
#     plt.savefig(os.path.join("results", "figures", "knowledge_base_feature_correlations_" + input_name + ".pdf"), format="pdf")
#     plt.clf()
#
#
# def plot_preprocessing_sampling_runtimes(df, input_name):
#     print("> Runtimes")
#     phases = ['Partitioning', 'Sampling']
#     subplot = 0
#     for i in [500]:  # [10,50,100,200]:
#         subplot += 1
#         plt.subplot(1, 4, subplot)
#         plt.title("Sample Size " + str(i))
#         subset_df = df.loc[df['sample_size'] == i]
#
#         t_partitioning_random = statistics.mean(subset_df.loc[subset_df['approach'] == 'Random']['time_partitioning'])
#         # t_alignment_random = statistics.mean(subset_df.loc[subset_df['approach'] == 'Random']['time_alignments'])
#         t_sampling_random = statistics.mean(subset_df.loc[subset_df['approach'] == 'Random']['time_sampling'])
#         t_sum = t_partitioning_random + t_sampling_random
#         t_list_random = [t_partitioning_random, t_sampling_random]
#         t_list_random_perc = [x / t_sum for x in t_list_random]
#         # print(t_list_random)
#
#         t_partitioning_longest = statistics.mean(subset_df.loc[subset_df['approach'] == 'Longest']['time_partitioning'])
#         # t_alignment_sequence = statistics.mean(subset_df.loc[subset_df['approach'] == 'Sequence']['time_alignments'])
#         t_sampling_longest = statistics.mean(subset_df.loc[subset_df['approach'] == 'Longest']['time_sampling'])
#         t_list_longest = [t_partitioning_longest, t_sampling_longest]
#         t_sum = t_partitioning_longest + t_sampling_longest
#         t_list_longest_perc = [x / t_sum for x in t_list_longest]
#         # print(t_list_longest)
#
#         t_partitioning_feature = statistics.mean(subset_df.loc[subset_df['approach'] == 'Feature']['time_partitioning'])
#         # t_alignment_feature = statistics.mean(subset_df.loc[subset_df['approach'] == 'Feature']['time_alignments'])
#         t_sampling_feature = statistics.mean(subset_df.loc[subset_df['approach'] == 'Feature']['time_sampling'])
#         t_list_feature = [t_partitioning_feature, t_sampling_feature]
#         t_sum = t_partitioning_feature + t_sampling_feature
#         t_list_feature_perc = [x / t_sum for x in t_list_feature]
#         # print(t_list_feature)
#
#         t_partitioning_sequence = statistics.mean(
#             subset_df.loc[subset_df['approach'] == 'Sequence']['time_partitioning'])
#         # t_alignment_sequence = statistics.mean(subset_df.loc[subset_df['approach'] == 'Sequence']['time_alignments'])
#         t_sampling_sequence = statistics.mean(subset_df.loc[subset_df['approach'] == 'Sequence']['time_sampling'])
#         t_list_sequence = [t_partitioning_sequence, t_sampling_sequence]
#         t_sum = t_partitioning_sequence + t_sampling_sequence
#         t_list_sequence_perc = [x / t_sum for x in t_list_sequence]
#         # print(t_list_sequence)
#
#         t_partitioning_combined = statistics.mean(
#             subset_df.loc[subset_df['approach'] == 'Combined']['time_partitioning'])
#         # t_alignment_sequence = statistics.mean(subset_df.loc[subset_df['approach'] == 'Sequence']['time_alignments'])
#         t_sampling_combined = statistics.mean(subset_df.loc[subset_df['approach'] == 'Combined']['time_sampling'])
#         t_list_combined = [t_partitioning_combined, t_sampling_combined]
#         t_sum = t_partitioning_combined + t_sampling_combined
#         t_list_combined_perc = [x / t_sum for x in t_list_sequence]
#         # print(t_list_combined)
#
#         # t_partitioning_sequence = statistics.mean(subset_df.loc[subset_df['approach'] == 'Sequence']['time_partitioning'])
#         ##t_alignment_sequence = statistics.mean(subset_df.loc[subset_df['approach'] == 'Sequence']['time_alignments'])
#         # t_sampling_sequence = statistics.mean(subset_df.loc[subset_df['approach'] == 'Sequence']['time_sampling'])
#         # t_list_sequence =[t_partitioning_sequence, t_alignment_sequence, t_sampling_sequence]
#         # t_sum = t_partitioning_sequence+t_alignment_sequence+t_sampling_sequence
#         # t_list_sequence_perc = [x / t_sum for x in t_list_sequence]
#         # print(t_list_sequence)
#
#         times_perc = pd.concat([pd.DataFrame({'Random': t_list_random_perc}, index=phases),
#                                 pd.DataFrame({'Longest': t_list_longest_perc}, index=phases),
#                                 pd.DataFrame({'Feature': t_list_feature_perc}, index=phases),
#                                 pd.DataFrame({'Sequence': t_list_sequence_perc}, index=phases),
#                                 pd.DataFrame({'Combined': t_list_combined_perc}, index=phases)],
#                                axis=1, sort=False)
#
#         times_perc.T.plot.bar(stacked=True)
#
#         plt.savefig(os.path.join("results", "figures", "runtimes_perc_" + input_name + ".pdf"), format="pdf")
#         plt.clf()
#
#         times_abs = pd.concat([pd.DataFrame({'Random': t_list_random}, index=phases),
#                                pd.DataFrame({'Longest': t_list_longest}, index=phases),
#                                pd.DataFrame({'Feature': t_list_feature}, index=phases),
#                                pd.DataFrame({'Sequence': t_list_sequence}, index=phases),
#                                pd.DataFrame({'Combined': t_list_combined}, index=phases)],
#                               axis=1, sort=False)
#
#         times_abs.T.plot.bar(stacked=True)
#
#         plt.savefig(os.path.join("results", "figures", "runtimes_" + input_name + ".pdf"), format="pdf")
#         plt.clf()
#
#
# def plot_correlation_distribution(df, input_name):
#     print("> Correlations")
#     subset_df = df.loc[df['informative'] == True]
#
#     my_order = subset_df.groupby(by=["feature"])["correlation"].median().iloc[::-1].index
#     sns.boxplot(x="feature", y="correlation", data=subset_df, order=my_order)
#
#     plt.savefig(os.path.join("results", "figures", "correlation_distribution_positive_" + input_name + ".pdf"), format="pdf")
#     plt.clf()
#
#
# def plot_partition_pruning(df, input_name):
#     print("> Pruning of Partitioning")
#     sns.barplot(x="p", y="total_size", hue="approach", data=df)
#     plt.savefig(os.path.join("results", "figures", "partition_sizes_different_p_" + input_name + ".pdf"), format="pdf")
#     plt.clf()
#
#
# def plot_partition_sizes(df, input_name):
#     print("> Partition Sizes")
#     for approach in ['Feature', 'Sequence', 'Combined']:
#         subset_df = df.loc[df['p'] == 5]
#         subset_df = subset_df.loc[subset_df['approach'] == approach]
#         sns.lineplot(x="partition", y="partition_size", data=subset_df)
#         plt.savefig(os.path.join("results", "figures", "partition_" + approach + "_" + input_name + ".pdf"), format="pdf")
#         plt.clf()


if __name__ == '__main__':
    main()
