import numpy as np
from itertools import product
from pandas import DataFrame as df
from tqdm import tqdm
from pandas import read_csv as rc


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
                      L_local,
                      L_global
                      ):
    """
    Input specifications (by example):
        forecast_horizon = 3, known as $k$ in our main work.
        Other choices include 1, 2, 5, 10.
        data_and_time. This should be a pandas dataframe including (1) a time index, (2) the dependent variable of interest
        and (3) exogenous variables.
        csv_directory. This is where the output will be exported to.
        functional_sets. This is the set of all models to be trained. It is obtained via the ModelGroupSpecs class.
        ar_orders = [1,..., 10]. Runs up to 5 for models in MG2T, MG3T, MG2V, MG3V and MG4V.
        window_sizes=[22, 63, 126, 252]
        L_Global.
    Return:
        - The set of all models produced by AL
        - The testing and training index
        - AL-induced optimal models
        - AL-induced optimal model forecasts
    """

    # Loading
    L_local_specification = L_local[0]

    v, Lambda = L_global

    if L_local_specification == 'Huber-Norm':
        lambda_vector = [Lambda**(v-i) for i in range(0, v)]
        quantile_C = L_local[1]

    elif L_local_specification == 'Ensemble-Huber':
        v0 = L_local[1]
        v1 = v-v0
        Ensemble_lambda_vector = [Lambda**(v1-i) for i in range(0, v1)]
        quantile_C = L_local[2]

    # Logistics: reading in the forecast dataframes.
    # The CSV files respectively contain [y_t|t-1, y_t|t-2, and y_t|t-3].

    Y_t1_given_t = rc(csv_directory + f'Y_t1_given_t.csv',
                      index_col=['time_index'])

    Y_t2_given_t = rc(csv_directory + f'Y_t2_given_t.csv',
                      index_col=['time_index'])

    Y_t3_given_t = rc(csv_directory + f'Y_t3_given_t.csv',
                      index_col=['time_index'])

    Y_t4_given_t = rc(csv_directory + f'Y_t4_given_t.csv',
                      index_col=['time_index'])

    Y_t5_given_t = rc(csv_directory + f'Y_t5_given_t.csv',
                      index_col=['time_index'])

    # Step 1: Housekeeping.
    # Step 1 (a):
    T_max = len(data_and_time) - 1
    W_max = max(window_sizes)
    P_max = max(ar_orders)

    # Step 1 (b): Define the training index.
    T_train = np.arange(W_max + 22 + P_max,
                        T_max - 22 + 1)

    # Step 1 (c): Define the testing index.
    T_test = np.arange(W_max + 100 + 44 + P_max,
                       T_max - 22 + 1)

    forecast_errors = create_H_tilda_dict(functional_sets)

    # Step 2 (a).
    for t in tqdm(T_train, leave=True, position=0):
        for model_name in functional_sets:

            # Step 2 (a)(i): Obtain the forecasts [y_t|t-3, y_t|t-2 and y_t|t-1].
            forecast_values = [Y_t1_given_t.loc[t-1, model_name],
                               Y_t2_given_t.loc[t-2, model_name],
                               Y_t3_given_t.loc[t-3, model_name],
                               Y_t4_given_t.loc[t-4, model_name],
                               Y_t5_given_t.loc[t-5, model_name],
                               ]

            # Step 2 (a)(ii): Obtain and save the forecasting error vector.
            val_data = [data_and_time[f'SPY_{k}d_returns'][t]]*k

            forecast_error = list(
                map(abs, np.subtract(forecast_values, val_data)))

            forecast_errors[model_name][t] = forecast_error

    # Step 3. Implement Adaptive Learning via the designated option.
    # Step 3 (b):

    # Step 3(b)(i): Define the local loss function
    # See below (1.7.2) for function declaration

    # Step 3(b)(ii): Declare T^{Huber}:
    T_Huber = np.arange(W_max + 45 + P_max,
                        T_max - 22 + 1)

    # Step 3(b)(iii): Iterate over T^{Huber} to find the constants for each component
    # of the error vector.
    C = calculate_quantiles(T_Huber,
                            forecast_errors,
                            functional_sets,
                            quantile_C,
                            k)

    v_t = create_v_t(functional_sets, T_Huber, forecast_errors, C, k)

    if L_local_specification == "Huber-Norm":
        optimal_models = {}
        y_star_dict = {}

    elif L_local_specification == "Ensemble-Huber":
        optimal_ensemble_weights = create_value_dict(functional_sets)
        ensembled_forecasts_all_t = {}
        v_norm_df = create_v_norm_df_Ensemble(
            v0, v1, functional_sets, T_test, Ensemble_lambda_vector, v_t)

    # Step 3(c).
    for t in tqdm(T_test, leave=True, position=0):

        # Step 3 (c)(i): Declare T_tilda (the adaptive learning lookback window).
        T_tilde = np.arange(t - v + 1, t + 1)

        if L_local_specification == "Huber-Norm":

            y_star, h_star = Huber_AL(T_tilde,
                                      t,
                                      functional_sets,
                                      Y_t5_given_t,
                                      v_t,
                                      lambda_vector)

            # Step 3 (d): Save this h star (best model) and make an associated forecast
            optimal_models.update({t: h_star})
            y_star_dict.update({t: y_star})

        elif L_local_specification == "Ensemble-Huber":

            ensembled_forecasts, weights = ensemble_Huber(t,
                                                          v0,
                                                          functional_sets,
                                                          Y_t5_given_t,
                                                          v_norm_df)

            # Save the ensemble weights and the ensemble forecasts
            optimal_ensemble_weights.update({t: weights})
            ensembled_forecasts_all_t.update({t: ensembled_forecasts})

    if L_local_specification == 'Ensemble-Huber':

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


###### 1.7.2 Learning Function with Huber Loss ######

# Step 3 (c)(iii)(2): Calculate the quantiles of the error distributions.

def calculate_quantiles(T_Huber, forecast_errors, functional_sets, quantile_C, k):
    C = {}
    for tau in T_Huber:
        # Step 3 (c)(iii)(2)(a): Collect the error vector for each model in the functional sets.
        quantile_set_C = []
        for i in range(k):
            E = [forecast_errors[model][tau][i] for model in functional_sets]
            quantile_set_C.append(np.quantile(E, quantile_C))
        C.update({tau: quantile_set_C})
    return C


def Scalar_Huber(x_i, C_i):
    if x_i >= C_i:
        return C_i*x_i - (1/2)*C_i**2
    else:
        return (1/2)*x_i**2


def Global_Huber(v, lambda_vector):
    return np.dot(v, lambda_vector)

###### 1.7.3 Ensemble ######


def Huber_AL(T_tilde,
             t,
             functional_sets,
             forecast_df,
             v,
             lambda_vector):

    loss_by_model = {}
    for model_name in functional_sets:

        # Step 3 (c)(iii)(A): Evaluate the global loss function,
        Huber_norms = [np.linalg.norm(v[model_name][tau], 1)
                       for tau in T_tilde]

        loss = Global_Huber(Huber_norms, lambda_vector)

        loss_by_model.update({model_name: loss})

    # Step 3 (c)(ii)(B): Find the argmin of the loss function for h in H over the period Tilda.
    h_star = min(loss_by_model, key=loss_by_model.get)

    # Step 3 (c)(ii)(C): Save this h star (best model) and make an associated forecast
    # Note: this forecast is y_{t+3|t}
    y_star = forecast_df.loc[t, h_star]

    return y_star, h_star


def ensemble_Huber(t, v0, functional_sets, forecast_df, v_norm_df):

    # Step 1: Declare T_0.
    T_0 = np.arange(t - v0 + 1, t+1)

    minimising_model_count = create_value_dict(functional_sets)

    # Re-write starts here
    # Step 2.
    for s in T_0:

        model_with_min_loss = v_norm_df.loc[s].idxmin(axis=1)

        minimising_model_count[model_with_min_loss] += 1

    # Re-write finishes here
    # Step 3: Calculate p^*_t as the empirical distribution of h^*_s.
    weights = {model: count/len(T_0)
               for model, count in minimising_model_count.items()}

    # Step 4: Produce and save the ensembled forecast and its associated ensemble weights.
    # Try removing loop and see if results are same. i.e. forecast_df.loc[t].
    forecasts_candidates = [
        np.array(forecast_df.loc[t, model]).transpose() for model in functional_sets]
    ensembled_forecasts = np.dot(list(weights.values()), forecasts_candidates)

    return ensembled_forecasts, weights


def create_v_t(functional_sets, T_Huber, abs_forecast_errors, C, k):

    v_t = create_H_tilda_dict(functional_sets)

    for tau in T_Huber:
        for model in functional_sets:
            vector_v = []
            for i in range(k):
                v_i = Scalar_Huber(
                    abs_forecast_errors[model][tau][i], C[tau][i])
                vector_v.append(v_i)

            v_t[model][tau] = vector_v

    return v_t


def create_v_norm_df_Ensemble(v0, v1, functional_sets, T_test, ensemble_lambda_vector, v_t):
    v_norm_df = df(np.arange(T_test[0] - v0 + 1,
                             T_test[-1] + 1), columns=['s'])
    v_norm_df.set_index(v_norm_df['s'], inplace=True)

    del v_norm_df['s']

    for s in tqdm(v_norm_df.index, leave=True, position=0):
        for model in functional_sets:

            T_1 = np.arange(s - v1 + 1, s + 1)
            v_norm_list = [np.linalg.norm(v_t[model][tau], 1)
                           for tau in T_1]
            v_norm_df.loc[s, model] = Global_Huber(
                v_norm_list, ensemble_lambda_vector)

    return v_norm_df

###### Helper Functions ######


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


def naming(model_group, ar_order, window_size, regressor, i, j):
    return f"(MG{model_group}, AR{ar_order}, W{window_size}, Regressor = {regressor}, i = {i}, j = {j})"
