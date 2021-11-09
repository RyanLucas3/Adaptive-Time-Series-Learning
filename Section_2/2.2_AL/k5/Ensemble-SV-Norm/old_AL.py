import numpy as np
from pandas import read_csv as rc
from tqdm import tqdm
from itertools import product


class ModelGroupSpecs:
    def __init__(self, data_and_time, ar_orders, window_sizes, desired_model_groups):
        self.ar_orders = ar_orders
        self.window_sizes = window_sizes
        self.data_and_time = data_and_time
        self.desired_model_groups = desired_model_groups

    def get_all_possible_combinations(self, model_group, MG_ar_orders, MG_window_sizes, MG_regressors, MG_i=[None], MG_j=[None]):
        models = list(product(model_group, MG_ar_orders,
                      MG_window_sizes, MG_regressors, MG_i, MG_j))
        return self.screen_models(models)

    def screen_models(self, models):
        screened_models = []
        for model in models:
            if (model[0] == '3V' or model[0] == '4V'):
                if 2*(3 + 9*model[1]) < model[2]:
                    screened_models.append(model)
            elif model[0] == '2V':
                if 2*(2 + 4*model[1]) < model[2]:
                    screened_models.append(model)
            else:
                screened_models.append(model)
        return screened_models

    def create_functional_sets(self):
        output = []
        if "MG1" in self.desired_model_groups:
            # MG1 = Model Group 1: AR models.

            self.MG1_model_specs = self.get_all_possible_combinations(
                model_group=['1'],
                MG_ar_orders=self.ar_orders,
                MG_window_sizes=self.window_sizes,
                MG_regressors=[None])
            output.append(self.MG1_model_specs)

        if "MG2N" in self.desired_model_groups:
            # MG2N = Model Group 2N: Single Variable Exogenous - VIX and YC term structure slopes.

            self.MG2N_model_specs = self.get_all_possible_combinations(
                model_group=['2N'],
                MG_ar_orders=[None],
                MG_window_sizes=self.window_sizes,
                MG_regressors=['VIX_slope', 'yc_slopes_3m_10y', 'yc_slopes_1m_10y', 'spread_3m_10y'])
            output.append(self.MG2N_model_specs)

        if "MG3N" in self.desired_model_groups:
            # MG3N = Model Group 3N: Multi-Variable regression on short and long term rates.

            self.MG3N_model_specs_VIX = self.get_all_possible_combinations(
                model_group=['3N'],
                MG_ar_orders=[None],
                MG_window_sizes=self.window_sizes,
                MG_regressors=[['vix_low', 'vix_high']],
                MG_i=['vix_spot', 'vix_1m', 'vix_2m', 'vix_3m'],
                MG_j=['vix_4m', 'vix_5m', 'vix_6m', 'vix_7m', 'vix_8m'])

            self.MG3N_model_specs_YC = self.get_all_possible_combinations(model_group=['3N'],
                                                                          MG_ar_orders=[
                                                                              None],
                                                                          MG_window_sizes=self.window_sizes,
                                                                          MG_regressors=[
                                                                              ['yc_low', 'yc_high']],
                                                                          MG_i=[
                                                                              'yc_1m', 'yc_3m', 'yc_6m', 'yc_1y', 'yc_2y'],
                                                                          MG_j=['yc_3y', 'yc_5y', 'yc_7y', 'yc_10y', 'yc_20y', 'yc_30y'])
            output.append(self.MG3N_model_specs_VIX)
            output.append(self.MG3N_model_specs_YC)

        if "MG2T" in self.desired_model_groups:
            # MG2T and MG3T: Introducing lagged dependent terms to the previous model specifications.

            self.MG2T_model_specs = self.get_all_possible_combinations(
                model_group=['2T'],
                MG_ar_orders=[1, 2, 3, 4, 5],
                MG_window_sizes=self.window_sizes,
                MG_regressors=['VIX_slope', 'yc_slopes_3m_10y', 'yc_slopes_1m_10y', 'spread_3m_10y'])
            output.append(self.MG2T_model_specs)

        if "MG3T" in self.desired_model_groups:

            self.MG3T_model_specs_VIX = self.get_all_possible_combinations(model_group=['3T'],
                                                                           MG_ar_orders=[
                                                                               1, 2, 3, 4, 5],
                                                                           MG_window_sizes=self.window_sizes,
                                                                           MG_regressors=[
                                                                               ['vix_low', 'vix_high']],
                                                                           MG_i=[
                                                                               'vix_spot', 'vix_1m', 'vix_2m', 'vix_3m'],
                                                                           MG_j=['vix_4m', 'vix_5m', 'vix_6m', 'vix_7m', 'vix_8m'])

            self.MG3T_model_specs_YC = self.get_all_possible_combinations(
                model_group=['3T'],
                MG_ar_orders=[1, 2, 3, 4, 5],
                MG_window_sizes=self.window_sizes,
                MG_regressors=[['yc_low', 'yc_high']],
                MG_i=['yc_1m', 'yc_3m', 'yc_6m', 'yc_1y', 'yc_2y'],
                MG_j=['yc_3y', 'yc_5y', 'yc_7y', 'yc_10y', 'yc_20y', 'yc_30y'])
            output.append(self.MG3T_model_specs_VIX)
            output.append(self.MG3T_model_specs_YC)

        if "MG2V" in self.desired_model_groups:
            # MG2V: VAR(p) models on VIX and Yield Curve Slopes.

            self.MG2V_model_specs = self.get_all_possible_combinations(
                model_group=['2V'],
                MG_ar_orders=[1, 2, 3, 4, 5],
                MG_window_sizes=self.window_sizes,
                MG_regressors=['VIX_slope', 'yc_slopes_3m_10y', 'yc_slopes_1m_10y', 'spread_3m_10y'])

            output.append(self.MG2V_model_specs)

        if "MG3V" in self.desired_model_groups:
            # MG3V: VAR(p) models on VIX and Yield short- and long-run rate pairs.

            self.MG3V_model_specs_VIX = self.get_all_possible_combinations(
                model_group=['3V'],
                MG_ar_orders=[1, 2, 3, 4, 5],
                MG_window_sizes=self.window_sizes,
                MG_regressors=[
                    ['vix_low', 'vix_high']],
                MG_i=[
                    'vix_spot', 'vix_1m', 'vix_2m', 'vix_3m'],
                MG_j=['vix_4m', 'vix_5m', 'vix_6m', 'vix_7m', 'vix_8m'])

            self.MG3V_model_specs_YC = self.get_all_possible_combinations(
                model_group=['3V'],
                MG_ar_orders=[1, 2, 3, 4, 5],
                MG_window_sizes=self.window_sizes,
                MG_regressors=[['yc_low', 'yc_high']],
                MG_i=['yc_1m', 'yc_3m', 'yc_6m', 'yc_1y', 'yc_2y'],
                MG_j=['yc_3y', 'yc_5y', 'yc_7y', 'yc_10y', 'yc_20y', 'yc_30y'])

            output.append(self.MG3V_model_specs_VIX)
            output.append(self.MG3V_model_specs_YC)

        if "MG4V" in self.desired_model_groups:
            # MG4V: VAR(p) models with Yield and VIX Curve Slopes pooled.

            self.MG4V_model_specs = self.get_all_possible_combinations(
                model_group=['4V'],
                MG_ar_orders=[1, 2, 3, 4, 5],
                MG_window_sizes=self.window_sizes,
                MG_regressors=[['Pooled Slopes']],
                MG_i=['VIX_slope'],
                MG_j=['yc_slopes_3m_10y', 'yc_slopes_1m_10y', 'spread_3m_10y'])

            output.append(self.MG4V_model_specs)

        if "MG2C" in self.desired_model_groups:
            # MG2C: cointegration models on VIX and Yield Curve Slopes.

            self.MG2V_model_specs = self.get_all_possible_combinations(
                model_group=['2C'],
                MG_ar_orders=[1, 2, 3, 4, 5],
                MG_window_sizes=self.window_sizes,
                MG_regressors=['VIX_slope', 'yc_slopes_3m_10y', 'yc_slopes_1m_10y', 'spread_3m_10y'])

            output.append(self.MG2V_model_specs)

        if "MG3C" in self.desired_model_groups:
            # MG3C: cointegration models on VIX and Yield short- and long-run rate pairs.

            self.MG3C_model_specs_VIX = self.get_all_possible_combinations(
                model_group=['3C'],
                MG_ar_orders=[1, 2, 3, 4, 5],
                MG_window_sizes=self.window_sizes,
                MG_regressors=[
                    ['vix_low', 'vix_high']],
                MG_i=[
                    'vix_spot', 'vix_1m', 'vix_2m', 'vix_3m'],
                MG_j=['vix_4m', 'vix_5m', 'vix_6m', 'vix_7m', 'vix_8m'],
                critical_values=[0.9, 0.95])

            self.MG3C_model_specs_YC = self.get_all_possible_combinations(
                model_group=['3C'],
                MG_ar_orders=[1, 2, 3, 4, 5],
                MG_window_sizes=self.window_sizes,
                MG_regressors=[['yc_low', 'yc_high']],
                MG_i=['yc_1m', 'yc_3m', 'yc_6m', 'yc_1y', 'yc_2y'],
                MG_j=['yc_3y', 'yc_5y', 'yc_7y', 'yc_10y', 'yc_20y', 'yc_30y'])

            output.append(self.MG3C_model_specs_VIX)
            output.append(self.MG3C_model_specs_YC)

        if "MG4C" in self.desired_model_groups:
            # MG4C: Cointegration models with Yield and VIX Curve Slopes pooled.

            self.MG4C_model_specs = self.get_all_possible_combinations(
                model_group=['4C'],
                MG_ar_orders=[1, 2, 3, 4, 5],
                MG_window_sizes=self.window_sizes,
                MG_regressors=[['Pooled Slopes']],
                MG_i=['VIX_slope'],
                MG_j=['yc_slopes_3m_10y', 'yc_slopes_1m_10y', 'spread_3m_10y'],
                critical_values=[0.9, 0.95])

            output.append(self.MG4C_model_specs)

        # Returning the functional sets to be deployed.
        return output

    def generate_H_tilda(self):
        H_tilda = []
        functional_sets = self.create_functional_sets()
        for functional_set in range(1, len(functional_sets) + 1):
            for model_group, ar_order, window_size, regressor, i, j in functional_sets[functional_set - 1]:
                H_tilda.append(naming(model_group, ar_order,
                                      window_size, regressor, i, j))
        return H_tilda


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

        forecast_horizon = 3, known as $k$ in our main work. 
        Other choices include 1, 2, 5, 10.

        data_and_time. This should be a pandas dataframe including (1) a time index, (2) the dependent variable of interest
        and (3) exogenous variables.

        csv_directory. This is where the output will be exported to. 

        functional_sets. This is the set of all models to be trained. It is obtained via the ModelGroupSpecs class.

        ar_orders = [1,..., 10]. For MG2T and MG3T these orders run to a maximum of 5.

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

    forecast_df = rc(csv_directory, index_col=['time_index'])

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
            forecast_value = forecast_df.loc[t, model_name]

            # Step 2 (a)(ii): Obtain the forecasting error.
            val_data = data_and_time[f'SPY_{k}d_returns'][t+k]

            error_t_plus_k = abs(forecast_value - val_data)

            forecast_errors[model_name][t+k] = error_t_plus_k

    # Step 3. Implement AL via the designated option.
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

            y_star, h_star = regular_AL(T_tilda=T_tilda,
                                        t=t,
                                        AL_specification="EN",
                                        functional_sets=functional_sets,
                                        forecast_df=forecast_df,
                                        forecast_errors=forecast_errors,
                                        lambda_vector=lambda_vector,
                                        p=p)

            print(y_star)

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

    # Step 3 (c)(ii).
    loss_by_model = {}

    for model_name in functional_sets:

        # Step 3 (c)(ii)(A): Collect the array of forecasting errors
        # over the adaptive learning lookback window and evaluate the loss over the period T tilda.
        errors = [forecast_errors[model_name][tau] for tau in T_tilda]

        if AL_specification == "EN":
            loss = exponential_learning(errors=errors, lam=lambda_vector, p=p)

        elif AL_specification == "Huber":
            loss = L_global(
                T_tilda, forecast_errors[model_name], lambda_vector, C1_dict, C2_dict)

        loss_by_model.update({model_name: loss})

    # Step 3 (c)(ii)(B): Find the argmin of the loss function for h in H over the period Tilda.
    h_star = min(loss_by_model, key=loss_by_model.get)

    # Step 3 (c)(ii)(C): Save this h star (best model) and make the associated forecast.
    y_star = forecast_df.loc[t, h_star]

    return y_star, h_star


###### 1.7.1 Exponential-Norm Learning Function ######

def exponential_learning(errors, lam, p):
    e_vector = np.array(errors)
    lambda_vector = np.array(lam)
    loss = np.dot(np.power(e_vector, p), lambda_vector)
    return loss

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
        x = abs(errors[tau])
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

###### 1.7.3 Ensemble ######


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
    weights = {model: count/len(T_0)
               for model, count in minimising_model_count.items()}

    # Step 4: Produce and save the ensembled forecast and its associated ensemble weights.
    forecasts_candidates = np.array(
        [np.array(forecast_df.loc[t, model]).transpose() for model in functional_sets])
    ensembled_forecasts = np.dot(
        np.array(list(weights.values())), forecasts_candidates)

    return weights, ensembled_forecasts

##### Helper Functions ####


def create_H_tilda_dict(H):
    H_tilda = {}
    for model in H:
        H_tilda.update({model: {}})
    return H_tilda


def naming(model_group, ar_order, window_size, regressor, i, j):
    return f"(MG{model_group}, AR{ar_order}, W{window_size}, Regressor = {regressor}, i = {i}, j = {j})"


def create_value_dict(H):
    H_tilda = {}
    for model in H:
        H_tilda.update({model: 0})
    return H_tilda
