# xai-series-CVD
 
# Explainable AI 
These files contain the explainable models tested. 

## Load and process dataset: 
```
Run the utils.py file 
drop the columns you desire to ommit from the analysis.

Run the kfold2.py to obtain results from the cross-validation. 
```
## Linear models: 
### These are interpretable models :
```
run 00_data_exploration.py file to obtain a descriptive analysis of the data and reproduce the results documented in the report 
```

## Linear models: 
### These are interpretable models:
```
run 01_interpretable_models.py to obtain the results of each of the glassbox visualisation of: 
- Logistic regression 
- Decision Tree
- Explainable boosting machine 
```
## Model agnostic methods: 
### These are methods used to explain blackbox models:
```
The blackbox model chosen as a basis for these methods is random forest. 
The first model-agnostic method is LIME: run 02_lime.py to obtain results. 
The second model-agnostic method is SHAP: run 02_shap.py to obtain results. 

```

## Saved Model outputs: 
### Pkl file:
```
The pkl file contains all outputs of the machine learning training saved in a .joblib or .pkl file.  

```

## Time: 
```
The time.py file was used to compare computation execution time of models.   

```