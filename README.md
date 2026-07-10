## Code Repo for Thesis CH24M571

1. **Project Overview** 

- **Problem Statement** : A Causal-Informed Framework for ETA-Aware Delivery Sequencing in Last-Mile Logistics
- **Objectives & Application Domain** : The objective is to design a city agnostic framework for delivery sequencing with optimising ETA as the primary goal. The domain for the thesis is Last Mile logistics.

---

2. **Programming Language & Framework** 

- **Languages used** : Python
- **Library & Frameworks** : 

    | Dependency | Version | Category / Usage |
    |------------|---------|------------------|
    | causal-learn | 0.1.4.7 | Core causal discovery (PC, CDNOD) |
    | tigramite | 5.2.10.1 | Time‑series causal discovery (PCMCI+) |
    | lightgbm | 4.6.0 | Oracle reward model estimation |
    | pandas | 3.0.3 | Data analysis and manipulation |
    | polars | 0.20.17 | High‑performance feature engineering |
    | duckdb | Latest | Querying dataset files in feature creators |
    | numpy | 2.5.0 | Numerical calculations |
    | scipy | 1.18.0 | Scientific computing |
    | scikit‑learn | 1.9.0 | Machine learning algorithms & evaluation |
    | matplotlib & seaborn | 3.11.0 / 0.13.2 | Visualization and plotting |
    | shap & joypy & xgboost | Latest | Graph plotting |
---
3. **Compute Resources**

- **CPU/GPU configuration** : The code is highly CPU bound process , GPU is not required . The code has been designed to run on Google Colab  & in local on a Octacore AMD Ryzen 7 AI 350 processor

- **RAM and storage** : The feature engineering pipeline is RAM hungry . Maximum Ram required is 16Gb and storage is 15Gb when all intermediate files are logged 

- **Cloud/local execution environment** : All codes are run as Colab files. Flags enabled for both local and colab runs . Local runs are optimised to run on machines with Ryzen Zen5 architecture that can use the execute 512-bit vector instructions to speed up parallel computations . Consequently , libraries are patched to use these instructions. Data is frequently read from & stored in Google drive. 

---

4. **Dataset Details**

- **Dataset source and size** :  [LaDe Dataset Huggingface ](https://huggingface.co/datasets/Cainiao-AI/LaDe)

 LaDe has a number of pickup & delivery datasets available out of which we utilise the following two files . 

- [Delivery Five cities (abbreviated as d5c) ](https://huggingface.co/datasets/Cainiao-AI/LaDe/blob/main/delivery_five_cities.csv) : Detailed delivery level logs over 1 month period [136 MB ]

- [20s sampled GPS logs](https://huggingface.co/datasets/Cainiao-AI/LaDe/blob/main/data_with_trajectory_20s/courier_detailed_trajectory_20s.pkl.xz) : Detailed log of GPS pings over 1 month period [439 MB compressed ]

- **Number of samples/classes/tokens** : The dataset involves 10,677k packages of 21k couriers and about 9 million GPS pings.Number of unique delivery person in Shanghai: 142 , 
Number of unique delivery person in Chongqing: 145 , 
Number of unique delivery person in Hangzhou: 178 

- **Preprocessing and train-test split** : Used Polars to manage large chunks of preprocessing pipelines taking advantage of its lazy-loading and zero copy features . Data is grouped into batches such that while splitting no batch spill happens . Data is splitted into 70:10:20 unless otherwise noted

---

5. **Tools and Execution Environment**

- **Processing engines/tools used** : Polars

---

6. **Code Organization**

- **Modular or single-file implementation** : Modular

- **Notebook-based or production-style structure** : Notebook

- **Folder hierarchy and configuration files** :

    Sequence of notebooks : Run the following notebooks sequentially

    - Run 06_weather.ipynb to generate weather files 
    - run 00_d5c_seperator.ipynb to pull the code that runs the city seperated delivery files 
    - Run 01_master_feature_creator.ipynb to generate features at batch level and delivery level 
    - Run 02_PC_delviery_level.ipynb to run PC algorithm at each individual city level
    - Run 03_PCMCI.ipynb to run pcmci at batch level : This confirms that batches are memory-less process 
    - Run 04_cdnod.ipynb to run the cdnod algorithm from 3 cities at 1000 samples each . This is a long running script - runs for about 18 hours due to N3 complexity 

---

7. **Pipeline Design**

- **Automatic or semi-automatic workflow** : Semi-automatic - Running all notebooks automatically would cause machines to run out of memory , hence the notebooks must be run sequentially

- **Data flow**: 

    Data visualisation -> Fetching weather data -> Pre-processing pipeline -> causal discovery -> reward oracle training -> Bandits training 

---

---





---
### Special Commands

While moving amongst local, github and colab , often the notebook gets corrupted while saving - "Invalid notebook ". This is an open bug in Colab . The following is the fix for that.  

- Create a backup first (optional but recommended) : 
```cp notebook.ipynb notebook_backup.ipynb```

- Use jq to delete the metadata.widgets key and save to a temporary file : 
`jq 'del(.metadata.widgets)' notebook.ipynb > temp_notebook.ipynb`

- Replace the original file with the fixed one : 
```mv temp_notebook.ipynb notebook.ipynb```