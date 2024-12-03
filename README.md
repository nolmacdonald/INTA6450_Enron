

<h1 align="center">INTA 6450 Enron Project</h1>

<div align="center">

<p align="center">
 <img height=150px src="./assets/enron.png" alt="Enron-logo">
</p>

<h4>INTA 6450 - Data Analytics and Security (Fall 2024)</h4>

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

</div>

## Overview

For the INTA 6450 Enron Project, this repository will host all the necessary information.
Any code development can be performed in a Jupyter Notebook or in a Python file, `src/file.py`.

__Enron Project Significant Files__:

- `notebooks/preprocessing.ipynb`: Pre-Processing steps before modeling
- `notebooks/topic_modeling.ipynb`: Topic model with scikit-learn using `from sklearn.decomposition import LatentDirichletAllocation`
- `notebooks/topic_modeling_gensim.ipynb`: Topic model using `gensim` and `pyLDAvis` (visualization)
- `notebooks/topic_modeling_portland.ipynb`: Topic model utilized in ___wrongdoing results___
- `notebooks/network_analysis.ipynb`: Network analysis modeling
- `src/data_wrangler.py`: Data parsing and wrangling class of utility functions
- `src/email_processing.py`: Pre-Processing class of utility functions
- `src/topic_model.py`: Topic modeling class with scikit-learn
- `src/utils/db_manager.py`: Database and data management functions
- `src/utils/log_config.py`: Logging class to log information, warnings, errors in other classes
- _OTHER_:
   - `data/models/lda_visualization.html`: LDA topic model plots and visualization presented in report/presentation

If the code is finalized, the code should be kept in a Python script in case of any changes or issues that can be identified. For example, you can keep the code in `src/fileName.py`.

General Repository Tree:
- `data`: Location where the Enron data base is stored after downloading from the [Enron Email Data Base (GT)](https://s3.amazonaws.com/inta.gatech/inta6450-emails.zip) (Data is not uploaded to the repository, ignored)
- `notebooks`: Any notebooks you worked on with code or markdown notes (`.ipynb` files)
- `src`: Main code including Python scripts after being developed in notebooks

*Note: There is LOTS of data, over 250K JSON files*

## Enron Instructions

In this portion of the project, you will actually implement a strategy to detect wrongdoing. 
You will start with several distinct proposals, 
so your group will work to harmonize them and come up with a single approach to implement as a group.

### Data

The Enron email dataset was released by FERC as part of their investigations into a number of executives. 
A dated description of the dataset is available here: [https://en.wikipedia.org/wiki/Enron_Corpus](https://en.wikipedia.org/wiki/Enron_Corpus)
You MAY approach this project with any version of the data set that you find online and find convenient. 
However, see the course [data access page](https://gatech.instructure.com/courses/402332/pages/data-access) 
for a hosted Elasticsearch server with accompanying example queries that can get you started. 

From the [data access page](https://gatech.instructure.com/courses/402332/pages/data-access), the Enron email is hosted on [Elasticsearch](https://www.elastic.co/elasticsearch/). The [Elasticsearch Host](http://18.188.56.207:9200/) is located at `18.188.56.207:9200`. To use queries in Elasticsearch, refer to [Elasticsearch Full Query DSL (Domain Specific Language)](https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl.html).

The Enron Corpus has been provided by the course instructor in a zipped file
at [Enron Email Data Base (GT)](https://s3.amazonaws.com/inta.gatech/inta6450-emails.zip).

### Goal

The goal of the project is to implement some kind of approach to filter through emails to identify the ones that are associated with your specific form of wrongdoing. You may choose to use methods from the course if you like but you do not need to. The most important part of a good project is that you have a clear type of wrongdoing laid out and that you can show that your methods do as good a job as possible at leading to emails that show your type of wrongdoing. The result of this project should be your identified emails that match your pattern and constitute evidence of your defined wrongdoing. You will be assessed on your ability to implement your project as well as how well your approach is appropriate for your definition of wrongdoing. 

### Process

1. You will be placed in a small group of four.  You may form your own groups if you like, but we will send out a survey early in the class to help make sure groups have people of different academic backgrounds but similar timezones.

2. Within your group, implement a strategy. It might or might not be the same as one of the strategies outlined in your individual Part 1 proposal. You may use the Python notebook at [data access page] as a start.

3. Find Top 5 emails representing signs of “wrongdoing”. It doesn't have to be exactly 5, but it almost certainly shouldn't be 50 and definitely shouldn't be 500!

4. Document how your strategy got your results. 

5. Write a 10 to 30-page paper, including appendixes/emails, that articulates the group’s wrongdoing target. The target doesn’t have to be the same as your proposal. The large range here is so that it can be OK to include differing amounts of long chunks of code or charts or email texts depending on your particular project choices.
    - Start with your definition of wrongdoing 
    - Indicate the strategy you used to find emails 
    - What is your set of emails that indicated wrongdoing?  
    - Was your set of emails good? Why or why not? 
    - What problems would you fix next time? 
    - What problems are unfixable about this approach? 

6. Assign one member of your group to submit the assignment to the appropriate section in Canvas.

### Rubric

| Criteria | Excellent (7.5 pts) | Good<br>  (3.75 pts) | Average (2.5 pts) | Below Avg.<br> (0 pts) | Pts |
|----------|-------------------|----------------|-----------------|-------------------|-----|
| Mechanics | No grammatical, spelling or punctuation errors. | Almost no grammatical, spelling or punctuation errors. | A few grammatical, spelling, or punctuation errors. | Many grammatical, spelling, or punctuation errors. | 7.5 |
| Completeness of Information | The paper included all required elements. | The paper included almost all the required elements. | The paper included most of the required elements. | The paper was missing many required elements. | 7.5 |
| Quality of Information | Your definition of wrongdoing is clear, logical, and practical. You have indicated the signs you are looking for to find relevant emails. You have implemented these strategies in Python or another computing framework. There is clear and insightful reflection on your chosen methods and their challenges. Example emails found are included and fit closely with your definition of wrongdoing. | You define wrongdoing with some clarity. You have indicated the signs of large groups of emails that you will read over. You have implemented these strategies in Python or another computing framework. There is reflection on your chosen methods and their challenges. Example emails found are included and are related to your definition of wrongdoing. | You define wrongdoing but there is ambiguity in your definition. Your signs of wrongdoing are coarse and implicate large numbers of emails about a range of topics. You have implemented these strategies in Python or another computing framework. There is reflection on your chosen methods and their challenges. Example emails found are included and are not strongly related to your definition of wrongdoing. | Definition of wrongdoing missing or very ambiguous. Signs of wrongdoing, implementation in any framework, and reflections are weak or missing. | 7.5 |
| Sequencing of Information | Information is organized in a clear, logical way. | Most information is organized in a clear, logical way. | Some information is logically sequenced. | There is no clear plan for the organization of information. | 7.5 |
| | | | | **Total Points:** | 30 |

## Installation

This section is a brief discussion if you need help with any installation or setup.

### Environment 

The first requirement is Python, which is recommended to be
installed with a virtual environment. For this example, the preferred Python installation will be through a Conda environment.
First you will need to install Conda, which we will install
[Conda Miniforge](https://github.com/conda-forge/miniforge). Conda Miniforge manages Python and Python packages but
only installs the minimal packages to reduce memory requirements.
[Installing Conda Miniforge documentation](https://docs.anaconda.com/miniconda/) discusses the installation process further.

With Conda installed, open your terminal and we will discuss some commands to enter. Note that `$` is not part of the command, just indicated the beginning of a new line. For example, we will create a Conda environment named `enron` to use.

### Easy Install

The easy way is to import the environment file stored in the repository to create an environment with package dependencies. The environment specifications are loaded from `env_enron.yml`. If you use the following command, you should be in the `INTA6450_Enron` directory.

```shell
$ conda env create -f env_enron.yml
```

This will create a conda environment named `enron` that can
be enabled by `conda activate enron`. The environment installs Python, all necessary packages for Jupyter Notebook (`notebook`, `nb_conda_kernels`, `ipython`), Matplotlib, NLTK, NumPy, Pandas and SciPy.

### Manual Install

Create a Conda environment named `enron`. After the environment is created, activate the environment so it can be utilized.
```shell
$ conda create -n enron
$ conda activate enron
```

Make sure that your environment has the Conda forge channel to install packages from the proper server. If the `conda-forge` channel does not appear, add the channel.
```shell
$ conda config --show channels
$ conda config --add channels conda-forge
```

Install your packages with `enron` activated. Note that if you did not activate the environment with `conda activate enron` it will install packages into your default environment in conda, `base`.
To install Python and the essentials to use Jupyter Notebook, use the following commands:
```
$ conda activate enron
$ conda install python notebook nb_conda_kernels ipython
```

If you want to install any other packages, just activate your environment and use `conda install packageName`.



