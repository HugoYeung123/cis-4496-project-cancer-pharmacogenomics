# Data and Feature Definitions

This document provides a central hub for the raw data sources, the processed/transformed data, and feature set sused in the Cancer Pharmacogenomics project. 

For each dataset, an individual report describing the data schema, the meaning of each data field, and other relevant contextual information is provided.  Where a dataset is the output of processing or feature engineering on existing datasets, the names of the input datasets and the scripts used to conduct those operations are noted. Sample datasets are stored in the project Data directory.

_**For ease of modifying this report, placeholder links are included in this page, for example a link to dataset 1, but they are just placeholders pointing to a non-existent page. These should be modified to point to the actual location.**_


## Raw Data Sources


| Dataset Name | Original Location   | Destination Location  | Data Movement Tools / Scripts | Link to Report |
| ---:| ---: | ---: | ---: | -----: |
| GDSC_dataset.csv | Sanger institute | Kaggle; GDSC | [script1.py](link/to/python/script/file/in/Code) |https://www.kaggle.com/datasets/samiraalipour/genomics-of-drug-sensitivity-in-cancer-gdsc  |
| genomic_features.csv | Sanger institue | cancerxgene.org | [script2.R](link/to/R/script/file/in/Code) | https://www.cancerrxgene.org/downloads/genetic_features?mutation=both|
| model_list_latest.csv | Sanger institue | sanger institute website | script2.R](link/to/R/script/file/in/Code) | [[https://www.cancerrxgene.org/downloads/genetic_features?mutation=both]](https://cellmodelpassports.sanger.ac.uk/downloads)(https://cog.sanger.ac.uk/cmp/download/model_list_20260303.csv) |


* Dataset1 summary. <Provide brief summary of the data, such as how to access the data. More detailed information should be in the Dataset1 Report.>
* Dataset2 summary. <Provide brief summary of the data, such as how to access the data. More detailed information should be in the Dataset2 Report.> 

## Processed Data
| Processed Dataset Name | Input Dataset(s)   | Data Processing Tools/Scripts | Link to Report |
| ---:| ---: | ---: | ---: | 
| Processed Dataset 1 | [Dataset1](link/to/dataset1/report), [Dataset2](link/to/dataset2/report) | [Python_Script1.py](link/to/python/script/file/in/Code) | [Processed Dataset 1 Report](link/to/report1)|
| Processed Dataset 2 | [Dataset2](link/to/dataset2/report) |[script2.R](link/to/R/script/file/in/Code) | [Processed Dataset 2 Report](link/to/report2)|

* Processed Data1 summary. <Provide brief summary of the processed data, such as why you want to process data in this way. More detailed information about the processed data should be in the Processed Data1 Report.>
* Processed Data2 summary. <Provide brief summary of the processed data, such as why you want to process data in this way. More detailed information about the processed data should be in the Processed Data2 Report.> 

## Feature Sets

| Feature Set Name | Input Dataset(s)   | Feature Engineering Tools/Scripts | Link to Report |
| ---:| ---: | ---: | ---: | 
| Feature Set 1 | [Dataset1](link/to/dataset1/report), [Processed Dataset2](link/to/dataset2/report) | [R_Script2.R](link/to/R/script/file/in/Code) | [Feature Set1 Report](link/to/report1)|
| Feature Set 2 | [Processed Dataset2](link/to/dataset2/report) |[SQL_Script2.sql](link/to/sql/script/file/in/Code) | [Feature Set2 Report](link/to/report2)|

* Feature Set1 summary. <Provide detailed description of the feature set, such as the meaning of each feature. More detailed information about the feature set should be in the Feature Set1 Report.>
* Feature Set2 summary. <Provide detailed description of the feature set, such as the meaning of each feature. More detailed information about the feature set should be in the Feature Set2 Report.> 
