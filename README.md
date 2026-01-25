# MODIFIED README FILE

## To get started

1. **Python Version:** Python 3.9 was used.
2. **Install Dependencies:**
```bash
pip install -r requirements.txt

```

3. **Download Data:**
Run the download script to fetch the event logs.
```bash
python download_data.py

```

4. **Prepare Data:**
Prepare the event logs for sampling (this discovers models and filters BPI 2018).
```bash
python prepare_data.py

```

5. **Run Experiments:**
Perform the sampling experiments.
```bash
python eval.py

```

6. **Generate Plots:**
Create the visual figures from your results.
```bash
python generate_plots.py

```

### Important Folders

* **`downloads/`** - Contains the raw downloaded event logs (output of `download_data.py`).
* **`data/`** - Contains the prepared models (`.pnml`) and filtered event logs (output of `prepare_data.py`).
* **`results/`** - Contains the sampling experiment results as CSVs (output of `eval.py`).
* **`results/figures/`** - Contains the generated PDF plots (output of `generate_plots.py`).
* **`results/ICPM_2021/`** - Contains the original results from the paper and the pre-computed **index files** (used by default in `eval.py`).

## Important Notes

* The inputs in `generate_plots.py` need to match the outputs in `results/`.

* **Modified `eval.py`:** The evaluation script has been modified from the original paper's implementation. Key changes include:
* **Configuration:** Uses a static `DATA_CONFIG` dictionary to manage paths explicitly.
* **Scope:** The "Full Log Analysis" (baseline) was removed to focus purely on sampling runtimes.
* **Error Handling:** Added try/except blocks to prevent the entire experiment from crashing if a single sample times out.

* **Data Sources:** `eval.py` uses the index files in `results/ICPM_2021/index_files/`. For the log files themselves, it loads the original logs from `downloads/` **except** for BPI 2018, which uses the filtered version in `data/`.
* **Overwriting Warning:** `eval.py` **overwrites existing result files** in `results/` every time it runs. If you have results you want to keep (e.g., a long run of BPI 2012), rename the file before restarting the script.

# BELOW IS THE ORIGINAL README FILE

# Relevance-guided Sampling for Conformance Checking

This repository contains an implementation, as well as the used experimental result data for the article ["Sampling What Matters: Relevance-guided Sampling of Event Logs"](https://ieeexplore.ieee.org/document/9576875).

The proposed sampling procedures guide the selection of traces to be included in a sample by learning which features of a trace correlate with trace properties of interest (so far we used deviating traces as interesting).

## How to use ##
### Using the command line interface ###
To generate a sample of the log from the command line, run:

```
python3 Logsampling.py <-index_file INDEX_FILE> <feature|behavioural> <sample size> <log_file> <model_file>
```

The parameters are
* ``algorithm`` - the indexing type to use. Can be either "feature" or "behavioural"
* ``sample size`` - the desired sample size
* ``log_file`` - the provided log file, in .xes format
* ``model_file`` - the model file used during sampling, in .pnml format
* ``index_file`` - (Optional)  the index file, containing the features to consider during sampling

### Invoking the classes from code ###
Alternatively, you can use the sampling procedures directly in your own project. 
For this, you first need to create the desired sampler:

#### Feature-based index
```python
sampler = FeatureGuidedLogSampler(log, index_file=index_file)
```
#### Sequence-based index
```python
sampler = SequenceGuidedLogSampler(log, batch_size=5, index_file=index_file)
```
Then, you may generate a sample by calling the following line:
```python
sample = sampler.construct_sample(log, model, initial_marking, final_marking, sample_size)
```

## Index files ##
By providing an index-file, you can specify, what log attributes may be considered as relevant during the generation of the sample.
In this xml-file you can specify the trace-level attributes, event-level attributes, k-gram lengths and potential preceeding discreditization steps, that comprise the feature index, that guides the sampling procedure.

The structure of the .xml-file is as follows
```xml
<index>
  <k>3</k>                                        #use activity k-grams of length 3
  <discretize>
    <attribute>attribute a</attribute>            #discretize trace-level attribute 'a' using a bucket size of 10
    <trace-level>true</trace-level>
    <bucket_width>10</bucket_width>
  </discretize>
    <discretize>
    <attribute>attribute b</attribute>             #discretize event-level attribute 'b' using a bucket size of 5
    <trace-level>false</trace-level>
    <bucket_width>5</bucket_width>
  </discretize>
  <discretize>                                    #additional attributes that need discretization
    ...
  </discretize>
  
  <features>
    <trace-level>
      <attribute>attribute a</attribute>          #incorporate trace-level attribue 'a'
      ...                                         #additional trace-level attributes to incorporate
    </trace-level>
    <event-level>
      <attribute>attribute b</attribute>          #incorporate event-level attribue 'b'
      ...                                         #additional event-level attributes to incorporate
    </event-level>
  </features>
</index>
```
If no index file is provided, all features without discretization, as well as 3-grams, are considered.

## Evaluation scripts and data ##
Under [results](https://github.com/MartinKabierski/Guided_Conformance_Sampling/tree/master/results), we provide the scripts used for experimental evaluation([eval.py](https://github.com/MartinKabierski/Guided_Conformance_Sampling/blob/MartinKabierski-patch-1/eval.py)) and plot generation([generate_plots.py](https://github.com/MartinKabierski/Guided_Conformance_Sampling/blob/MartinKabierski-patch-1/generate_plots.py)), as well as the .csv-files and plots, that were generated using these scripts and used for the article.


## Citing this work ##
If you use this repository, please cite the article:
```
@INPROCEEDINGS{9576875,
  author={Kabierski, Martin and 
    Nguyen, Hoang Lam and 
    Grunske, Lars and 
    Weidlich, Matthias},
  booktitle={2021 3rd International Conference on Process Mining (ICPM)}, 
  title={Sampling What Matters: Relevance-guided Sampling of Event Logs}, 
  year={2021},
  volume={},
  number={},
  pages={64-71},
  doi={10.1109/ICPM53251.2021.9576875}}
}
```
