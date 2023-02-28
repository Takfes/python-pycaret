from pycaret.datasets import get_data
from pycaret.regression import RegressionExperiment

# * https://pycaret.readthedocs.io/en/latest/api/regression.html#pycaret.regression.add_metric
# * https://medium.com/@moez-62905/pycaret-3-is-coming-whats-new-8d7d241c40d8
# * https://towardsdatascience.com/5-things-you-are-doing-wrong-in-pycaret-e01981575d2a

# TODO how to add custom metric
# TODO how to add custom pipeline
# TODO error when set logging_param = True


data = get_data("insurance")

# what is going on w/ datatypes
data.dtypes
data.shape
# data2 = data.copy()
# data2['smoker'] = data2['smoker'].astype('category')
# data2.dtypes
# data2.region.nunique()
# data2.region.unique()

exp1 = RegressionExperiment()
exp1.setup(data, target="charges", session_id=123)

attrs = [x for x in dir(exp1) if not x.startswith("_")]
attrs

# Customer scoring function
# from sklearn.metrics import make_scorer
# import numpy as np

# Define a function to calculate sMAPE
# def smape(y_true, y_pred):
#     """
#     Calculate symmetric mean absolute percentage error (sMAPE)
#     """
#     numerator = np.abs(y_pred - y_true)
#     denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
#     return np.mean(numerator / denominator) * 100

# # Define the custom scorer using make_scorer
# smape_scorer = make_scorer(smape, greater_is_better=False)

# def symmetric_mape(y_true, y_pred):
#     return np.mean(np.abs((y_true - y_pred) / (y_true + y_pred))) * 100

# # define wrapper function to add name attribute
# def make_named_scorer(scorer_func):
#     def named_scorer_func(y_true, y_pred):
#         return scorer_func(y_true, y_pred)
#     named_scorer_func.__name__ = scorer_func.__name__
#     return named_scorer_func

# symmetric_mape_scorer = make_scorer(make_named_scorer(symmetric_mape))
# exp1.add_metric(id = 'smape', name = 'SMAPE',score_func = symmetric_mape_scorer)

exp1.get_metrics()
exp1.models()

# Check inferred datatypes
exp1.get_config("X_train").dtypes

# Check pipeline
exp1.get_config("pipeline")

# Check and edit configurations
exp1.get_config("logging_param")
exp1.get_config("exp_name_log")

# exp1.set_config("logging_param", True)
# exp1.set_config("exp_name_log", "insurance_data")

# Some auto-eda
exp1.eda()

# train models
best_model = exp1.compare_models(n_select=5)
huber = exp1.create_model("huber")
best_model.append(huber)

# ensemble model
ens_gbr = exp1.ensemble_model(best_model[0], choose_better=True)
best_model.append(ens_gbr)
len(best_model)

# tune models
# best_model[0] is ens_gbr
tuned_models = [exp1.tune_model(m) for m in best_model[:-1]]
tuned_models.append(ens_gbr)
len(tuned_models)

tuned_models2 = [exp1.tune_model(m, search_library="optuna") for m in best_model[:-1]]

# blend models
blended_models = exp1.blend_models(tuned_models, choose_better=True)

# stack models
stacked_models = exp1.stack_models(tuned_models, choose_better=True)

# evaluate model
# exp1.evaluate_model(stacked_models)

exp1.predict_model(blended_models)
exp1.predict_model(stacked_models)

fm1 = exp1.finalize_model(blended_models)
fm2 = exp1.finalize_model(stacked_models)

# best out of all models in the session
fmb = exp1.automl(optimize="MAE")

# save model
exp1.save_model(fm1, "fm1")
exp1.save_model(fm2, "fm2")

# create api
exp1.create_api(fm1, "fm1_api")

# create docker
exp1.create_docker("fm1_api")
