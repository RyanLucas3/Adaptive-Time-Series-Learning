from numpy.lib.function_base import quantile
import numpy as np
from numpy.random.mtrand import exponential
from pandas import read_csv as rc
from pandas import DataFrame as df
from tqdm import tqdm
import cProfile
import pstats
import io
import glob


def profile(function):

    def inner(*args, **kwargs):

        pr = cProfile.Profile()
        pr.enable()
        retval = function(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return retval

    return inner


def adaptive_learning(k,
                      data_and_time,
                      csv_directory,
                      functional_sets,
                      ar_orders,
                      window_sizes,
                      specification_learning
                      ):
    """
    Input specifications (by example):

        forecast_horizon=3, known as $k$ in our main work.
            Choices include 5, 10, 15, 22.

        data_and_time. This should be a pandas dataframe including (1) a time index, (2) the dependent variable of interest
        and (3) exogenous variables.

        csv_directory. This is where the output will be exported to.

        functional_sets. This is the set of all models to be trained. It is obtained via the ModelGroupSpecs class.

        window_sizes=[22, 63, 126, 252]

        specification_learning:

            E.g. ['EN',[100,1,1]]
                 ['Huber',[100,1,1],[0.25,0.5]]
                 ['Ensemble_EN',[100,1,1],[30]]

    Return:

        - The set of all models produced by AL
        - The testing and training index
        - AL-induced optimal models
        - AL-induced optimal model forecasts

    """

    # Loading
    AL_specification = specification_learning[0]

    if AL_specification == 'EN':
        v, p, Lambda = specification_learning[1]
        lambda_vector = [Lambda**(v-i) for i in range(0, v)]

    elif AL_specification == 'Huber':
        v, p, Lambda = specification_learning[1]
        quantile_C1, quantile_C2 = specification_learning[2]
        lambda_vector = [Lambda**(v-i) for i in range(0, v)]

    elif AL_specification == 'Ensemble_EN':
        v, p, Lambda = specification_learning[1]
        v0 = specification_learning[2][0]
        v1 = v-v0
        lambda_vector = [Lambda**(v1-i) for i in range(0, v1)]

    """
        NOT IN USE CURRENTLY
        elif AL_specification=='Ensemble_Huber'
    """

    # Reading in the forecasts.
    forecast_dfs = []
    path = f"{csv_directory}/**.csv"
    for fname in glob.glob(path):
        dataframe = rc(fname, index_col=['time_index'])
        forecast_dfs.append(dataframe)

    forecast_df = rc(
        '../../DataCentre/AdaptiveLearning/Real_data/Forecasts/k3/inter_forecasts_k3_0_1.csv', index_col=['time_index'])

    # Step 1: Housekeeping.
    # Step 1 (a):
    T_max = len(data_and_time) - 1
    W_max = max(window_sizes)
    P_max = max(ar_orders)

    # Step 1 (b): Define the training index.
    T_train = np.arange(W_max + 22 + P_max,
                        T_max - 22 + 1)

    # Step 1 (c): Define the testing index.
    T_test = np.arange(W_max + 44 + 100 + P_max,
                       T_max - 22 + 1)

    forecast_errors = create_H_tilda_dict(functional_sets)

    # Step 2.
    for t in tqdm(T_train, leave=True, position=0):
        for model_name in functional_sets:

            # Step 2 (a)(i): Obtain a forecast according to the model.
            forecast_values = []
            for i in range(k):
                forecast_values.append(forecast_dfs[i].loc[t, model_name])

            # Step 2 (a)(ii): Obtain the forecasting error.
            val_data = [data_and_time[f'SPY_{k}d_returns'][t+k]]*3

            error_t_plus_k = np.linalg.norm(x=np.subtract(
                np.array(forecast_values), np.array(val_data)), ord=p)

            forecast_errors[model_name][t+k] = error_t_plus_k

    # Step 3.
    if AL_specification == 'Huber':

        # Step 3(b)(i): Define the local loss function
        # See below (1.7.2) for function declaration

        # Declare T_Huber:
        T_Huber = np.arange(W_max + 45 + P_max,
                            T_max - 22 + 1)

        # Step 3(b)(ii): Find constants
        C1, C2 = calculate_quantiles(T_Huber,
                                     forecast_errors,
                                     functional_sets,
                                     quantile_C1,
                                     quantile_C2)

        optimal_models = {}
        y_star_dict = {}

    elif AL_specification == "EN":
        optimal_models = {}
        y_star_dict = {}

    elif AL_specification == "Ensemble_EN":
        optimal_ensemble_weights = create_value_dict(functional_sets)
        ensembled_forecasts_all_t = {}

    # Step 3(c).
    for t in tqdm(T_test, leave=True, position=0):

        # Step 3 (c)(i): Declare T_tilda (the adaptive learning lookback window).
        T_tilda = np.arange(t - v + 1, t + 1)

        if AL_specification == "EN":

            h_star, y_star = regular_AL(T_tilda=T_tilda,
                                        t=t,
                                        AL_specification="EN",
                                        functional_sets=functional_sets,
                                        forecast_df=forecast_df,
                                        forecast_errors=forecast_errors,
                                        lambda_vector=lambda_vector,
                                        p=p)

            # Step 3 (d): Save this h star (best model) and make an associated forecast
            optimal_models.update({t: h_star})
            y_star_dict.update({t: y_star})

        elif AL_specification == "Ensemble_EN":

            weights, ensembled_forecasts = ensemble_EN_AL(
                t=t,
                v0=v0,
                v1=v1,
                p=p,
                lambda_vector=lambda_vector,
                functional_sets=functional_sets,
                forecast_df=forecast_df,
                forecast_errors=forecast_errors)

            # Save the ensemble weights and the ensemble forecasts
            optimal_ensemble_weights.update({t: weights})
            ensembled_forecasts_all_t.update({t: ensembled_forecasts})

        elif AL_specification == "Huber":

            y_star, h_star = regular_AL(T_tilda=T_tilda,
                                        t=t,
                                        AL_specification="Huber",
                                        functional_sets=functional_sets,
                                        forecast_df=forecast_df,
                                        forecast_errors=forecast_errors,
                                        lambda_vector=lambda_vector,
                                        p=p,
                                        C1_dict=C1,
                                        C2_dict=C2)

            # Step 3 (d): Save this h star (best model) and make an associated forecast
            optimal_models.update({t: h_star})
            y_star_dict.update({t: y_star})

    if AL_specification == 'Ensemble_EN':

        Output = [
            functional_sets,
            [T_test, T_train],
            optimal_ensemble_weights,
            ensembled_forecasts_all_t
        ]

    else:
        Output = [
            functional_sets,
            [T_test, T_train],
            optimal_models,
            y_star_dict
        ]

    return Output


def regular_AL(T_tilda,
               t,
               AL_specification,
               functional_sets,
               forecast_df,
               forecast_errors,
               lambda_vector,
               p,
               C1_dict=None,
               C2_dict=None):

    # Step 3 (b): For all functional sets in our total model set.
    loss_by_model = {}

    for model_name in functional_sets:

        # Step 3 (b)(i) Collect the array of forecasting errors
        # over the adaptive learning lookback window.
        errors = [forecast_errors[model_name][tau] for tau in T_tilda]

        # # Step 3 (b)(ii): Evaluate the loss over the period T tilda.
        if AL_specification == "EN":
            loss = exponential_learning(errors=errors, lam=lambda_vector, p=p)

        elif AL_specification == "Huber":
            loss = L_global(
                T_tilda, forecast_errors[model_name], lambda_vector, C1_dict, C2_dict)

        loss_by_model.update({model_name: loss})

    # # Step 3 (c): Find the argmin of the loss function for h in H over the period Tilda.
    h_star = min(loss_by_model, key=loss_by_model.get)

    # Step 3 (d): Save this h star (best model) and make an associated forecast
    y_star = forecast_df.loc[t, h_star]

    return y_star, h_star


def ensemble_EN_AL(t, v0, v1, p, lambda_vector, functional_sets, forecast_df, forecast_errors):

    # Step 1: Declare T_0.

    T_0 = np.arange(t - v0 + 1, t+1)

    minimising_model_count = create_value_dict(functional_sets)

    # Step 2.
    for s in T_0:

        # Step 2 (a): Declare T_1.
        T_1 = np.arange(s - v1 + 1, s+1)

        loss_by_model = create_value_dict(functional_sets)

        # For all h in H.
        for model_name in functional_sets:

            # Step 2 (b): Collect the array of forecasting errors for all models
            # over the adaptive learning lookback window.
            errors = [forecast_errors[model_name][tau] for tau in T_1]

            # # Step 2 (c): Evaluate the loss over the period T_1.
            loss = exponential_learning(errors=errors, lam=lambda_vector, p=p)
            loss_by_model.update({model_name: loss})

        # Step 2 (c): Obtain h*_s.
        # The argmin of loss at time s.
        model_with_min_loss = min(loss_by_model, key=loss_by_model.get)
        minimising_model_count[model_with_min_loss] += 1

    # Step 3: Calculate p^*_t as the empirical distribution of h^*_s.
    # INEFFICIENT
    weights = {model: count/len(T_0)
               for model, count in minimising_model_count.items()}

    # Step 4: Produce and save the ensembled forecast and its associated ensemble weights.
    forecasts_candidates = np.array([np.array(forecast_df.loc[t, model]).transpose(
    ) for model in functional_sets])
    ensembled_forecasts = np.dot(
        np.array(list(weights.values())), forecasts_candidates)  # INEFFICIENT

    return weights, ensembled_forecasts

###### 1.7.2 Local Loss function: Huber ######


def L_local(x, C1, C2):
    if x >= C2:
        return (C2 - C1)*(x) + (C1**2 - C2**2)/2
    elif (C2 > x and x > C1):
        return ((x - C1)**2)/2
    else:
        return 0


def L_global(T_tilda, errors, lambda_vector, C1_dict, C2_dict):
    local_loss = []
    for tau in T_tilda:
        x = errors[tau]
        local_loss.append(L_local(x, C1_dict[tau], C2_dict[tau]))
    return np.dot(local_loss, lambda_vector)


def calculate_quantiles(T_Huber, forecast_errors, functional_sets, quantile_C1, quantile_C2):
    C1 = {}
    C2 = {}
    for tau in T_Huber:
        E = [forecast_errors[model][tau] for model in functional_sets]
        C1.update({tau: np.quantile(E, quantile_C1)})
        C2.update({tau: np.quantile(E, quantile_C2)})
    return C1, C2

################################################


def exponential_learning(errors, lam, p):
    return np.dot(errors, lam)


def create_H_tilda_dict(H):
    H_tilda = {}
    for model in H:
        H_tilda.update({model: {}})
    return H_tilda


def create_value_dict(H):
    H_tilda = {}
    for model in H:
        H_tilda.update({model: 0})
    return H_tilda
