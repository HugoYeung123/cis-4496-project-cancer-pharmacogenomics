# Baseline Model Report

_Baseline model is the the model a data scientist would train and evaluate quickly after he/she has the first (preliminary) feature set ready for the machine learning modeling. Through building the baseline model, the data scientist can have a quick assessment of the feasibility of the machine learning task._

When applicable, the Automated Modeling and Reporting utility developed by TDSP team of Microsoft is employed to build the baseline models quickly. The baseline model report is generated from this utility easily. 

> If using the Automated Modeling and Reporting tool, most of the sections below will be generated automatically from this tool. 

## Analytic Approach
* What is target definition
* What are inputs (description)
* What kind of model was built?

## Model Description

* Models and Parameters

	* Description or images of data flow graph
  		* if AzureML, link to:
    		* Training experiment
    		* Scoring workflow
	* What learner(s) were used?
	* Learner hyper-parameters


## Results (Model Performance)
* ROC/Lift charts, AUC, R^2, MAPE as appropriate
* Performance graphs for parameters sweeps if applicable

  
 Model 				      RMSE		  MAE		 R2
XGBoost(w/ Optuna)   	1.032373	0.768512   0.8603
LightGBM(w/ Optuna) 	1.035314  	0.772103   0.859503
Catboost(w/ Optuna) 	1.050988 	0.784145   0.855216
HistGradient(w/ Optuna) 1.099774	0.821224   0.841463
Catboost				1.110923    0.829804   0.838232
Extra Trees				1.2494		0.9181	   0.7954
Ridge(w/ Optuna)		1.341		1.006	   0.76426895
Linear Regression		1.3443		1.0079	   0.7631
HistGradient			1.396169	1.047938   0.741699
MLPRegressor			1.409077	1.059455   0.7369

## Model Understanding

* Variable Importance (significance)
  - The high cardinality feature, which was separately encoded as **DRUG_NAME_target_enc** is considered to be the biggest predictor in the baseline feature analysis, showing that identity carried the most weight when it came to the prediction for **LN_IC50**.
  - The merged numerical features of **ploidy_wes**, **ploidy_snp6**, and **mutational burden** were among the most informative features for the baseline models.
  - Other Cancer-context based features like **TCGA_DESC**, **GDSC_Tissue_descriptor_1**, **GDSC_Tissue_descriptor_2**, and **TARGET_PATHWAY** contributed to the model by suporting the signals that account for the tissue, disease-specific, and pathway response patterns.
* Insight Derived from the Model



## Conclusion and Discussions for Next Steps

* Conclusion on Feasibility Assessment of the Machine Learning Task

* Discussion on Overfitting (If Applicable)

* What other Features Can Be Generated from the Current Data

* What other Relevant Data Sources Are Available to Help the Modeling
