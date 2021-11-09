import numpy as np
from itertools import product
from pandas import DataFrame as df
from statsmodels.tsa.vector_ar.var_model import forecast
from tqdm import tqdm
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import VAR
from pandas import read_csv as rc
import pickle


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


def forecasting(k,
                data_and_time,
                functional_sets,
                ar_orders=list(range(1, 11)),
                window_sizes=[22, 63, 126, 252]
                ):
    """
    Input specifications (by example):
        forecast_horizon=3, known as $k$ in our main work.
        Other choices include 1, 2, 5, 10.
        data_and_time. This should be a pandas dataframe including (1) a time index, (2) the dependent variable of interest
        and (3) exogenous variables.
        functional_sets. This is the set of all models to be trained. It is obtained via the ModelGroupSpecs class.
        ar_orders = [1, 2,..., 10]. For MG2T and MG3T these order run to up to 5.
        window_sizes = [22, 63, 126, 252].

    Return:
        - The set of forecasts produced by all fixed model groups.
    """

    # Step 1 (a): Defining our indexes for training and testing.
    T_max = len(data_and_time) - 1
    W_max = max(window_sizes)
    P_max = max(ar_orders)
    T_train = np.arange(W_max + 19 + P_max, T_max - 22 + 1)
    # Logistics
    # Creation of H_tilda

    forecast_df = create_forecast_df(T_train, functional_sets)

    parameters = {
        "(MG3T, AR4, W252, Regressor = ['vix_low', 'vix_high'], i = vix_3m, j = vix_8m)": {}, "(MG2N, ARNone, W63, Regressor = VIX_slope, i = None, j = None)": {}}

    # Step 2 (a): Training the models and making a forecast.
    for t in tqdm(T_train, leave=True, position=0):

        forecasts_t = [data_and_time['Date'][t]]

        for functional_set in functional_sets:
            for model_group, ar_order, window_size, regressor, i, j in functional_set:

                model_name = naming(model_group, ar_order,
                                    window_size, regressor, i, j)

                if model_name in parameters.keys():
                    if model_group == '1':

                        forecasts = train_and_forecast_MG1(data_and_time[f'SPY_{k}d_returns'],
                                                           t,
                                                           window_size,
                                                           ar_order,
                                                           k)

                    elif model_group == '2N':

                        forecasts = train_and_forecast_MG2N(data_and_time[regressor],
                                                            data_and_time[f'SPY_{k}d_returns'],
                                                            t,
                                                            window_size,
                                                            k,
                                                            model_name,
                                                            parameters)

                    elif model_group == '3N':

                        high_low_point = np.column_stack([data_and_time[i],
                                                          data_and_time[j]])

                        forecasts = train_and_forecast_MG3N(high_low_point,
                                                            data_and_time[f'SPY_{k}d_returns'],
                                                            t,
                                                            window_size,
                                                            k)

                    elif model_group == '2T':

                        forecasts = train_and_forecast_MG_2T_3T(dep_data=data_and_time[f'SPY_{k}d_returns'],
                                                                regressor=data_and_time[regressor],
                                                                t=t,
                                                                w=window_size,
                                                                ar_order=ar_order,
                                                                k=k)

                    elif model_group == '3T':

                        high_low_point = np.column_stack([data_and_time[i],
                                                          data_and_time[j]])

                        forecasts = train_and_forecast_MG_2T_3T(dep_data=data_and_time[f'SPY_{k}d_returns'],
                                                                regressor=high_low_point,
                                                                t=t,
                                                                w=window_size,
                                                                ar_order=ar_order,
                                                                k=k,
                                                                param_dict=parameters,
                                                                model_name=model_name)

                    elif model_group == '2V':

                        data = np.column_stack([data_and_time[f'SPY_{k}d_returns'],
                                                data_and_time[regressor]])

                        forecasts = train_and_forecast_VAR(data,
                                                           t,
                                                           window_size,
                                                           ar_order,
                                                           k)
                    elif model_group == '3V':

                        data = np.column_stack([data_and_time[f'SPY_{k}d_returns'],
                                                data_and_time[i],
                                                data_and_time[j]])

                        forecasts = train_and_forecast_VAR(data,
                                                           t,
                                                           window_size,
                                                           ar_order,
                                                           k)
                    elif model_group == '4V':

                        data = np.column_stack([data_and_time[f'SPY_{k}d_returns'],
                                                data_and_time[i],
                                                data_and_time[j]])

                        forecasts = train_and_forecast_VAR(data,
                                                           t,
                                                           window_size,
                                                           ar_order,
                                                           k)

                forecasts_t.append(forecasts)

        forecast_df.loc[t] = forecasts_t

    with open('/Users/ryanlucas/Desktop/pickle/params.pickle', 'wb') as handle:
        pickle.dump(parameters, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)


##### Individual Training and forecasting Functions See Appendix 1.5 and 1.6 of the pseudo-algorithm #####

def train_and_forecast_MG1(data, t, w, ar_order, k):

    windowed_data = np.array(data[(t - w + 1): (t + 1)])

    model = ARIMA(windowed_data, order=(
        ar_order, 0, 0)).fit(method="yule_walker")

    forecasts = model.forecast(k)[::-1]

    return forecasts


def train_and_forecast_MG2N(regressor, dep_data, t, w, k, model_name, param_dict):

    forecasts = []

    indep_train_data = np.array(regressor[(t - w - k + 1): (t - k + 1)])

    indep_train_data = sm.add_constant(indep_train_data)

    dep_train_data = np.array(dep_data[(t - w + 1): (t + 1)])

    model = sm.OLS(dep_train_data,
                   indep_train_data).fit()

    if model_name in param_dict.keys():
        param_dict[model_name][t] = model.params

    for i in range(k):

        forecasts.append(model.params[0] + model.params[1] * regressor[t - i])

    return forecasts


def train_and_forecast_MG3N(regressor, dep_data, t, w, k):

    forecasts = []

    indep_train_data = regressor[(t - w - k + 1): (t - k + 1)]

    indep_train_data = sm.add_constant(indep_train_data)

    dep_train_data = dep_data[(t - w + 1): (t + 1)]

    model = sm.OLS(dep_train_data,
                   indep_train_data).fit()

    for i in range(k):

        forecasts.append(
            np.dot(model.params, sm.add_constant(regressor)[t - i]))

    return forecasts


def train_and_forecast_MG_2T_3T(dep_data, regressor, t, w, ar_order, k, param_dict, model_name):

    indep_train_data = vectorise_indep_variables(dep_data,
                                                 exog=regressor,
                                                 t=t,
                                                 ar_order=ar_order,
                                                 w=w,
                                                 k=k)

    dep_train_data = dep_data[(t - w + 1): (t + 1)]

    model = sm.OLS(dep_train_data, sm.add_constant(indep_train_data)).fit()

    if model_name in param_dict.keys():
        param_dict[model_name][t] = model.params

    forecasts = forecast_with_lags(
        model.params, dep_data, regressor, t, ar_order, k)

    return forecasts


def train_and_forecast_VAR(data, t, w, ar_order, k):

    windowed_data = data[(t - w + 1): (t + 1)]

    model = VAR(windowed_data).fit(ar_order)

    forecasts = model.forecast(
        y=data[(t - (ar_order) + 1): (t + 1), :], steps=k)[:, 0]

    return forecasts[::-1]


def vectorise_indep_variables(dep_to_be_lagged, exog, t, ar_order, w, k):

    lagged_dep_data = list(dep_to_be_lagged[(t - w - ar_order): (t)])

    lagged_dep_data = lagged_dep_data[::-1]

    lagged_p = [lagged_dep_data[j: j+ar_order]
                for j in range(0, len(lagged_dep_data) - ar_order)]

    lagged_p = np.array(lagged_p[:: -1])

    exog = np.array(exog[(t - w - k + 1): (t - k + 1)])

    vectorised_training_data = np.array(
        [np.append(lagged_p[j], [exog[j]]) for j in range(0, len(exog))])

    return vectorised_training_data


def forecast_with_lags(model_params, dep_to_be_lagged, regressor, t, ar_order, k):

    forecasts = []

    lagged_y = np.array(
        dep_to_be_lagged[(t - ar_order + 1): (t + 1)])
    lagged_x = np.array(regressor[(t - k + 1): (t + 1)])

    for tau in range(0, k):  # tau = 0 means 1-step-ahead.

        observations_under_consideration = [[1]]

        if len(forecasts) > 0:
            observations_under_consideration.append(
                forecasts[::-1][:ar_order])

        observations_under_consideration.append(lagged_y[tau:][::-1])

        observations_under_consideration.append(lagged_x[tau].flatten())

        observations_under_consideration = np.concatenate(
            observations_under_consideration).ravel()

        forecasts.append(np.dot(np.array(observations_under_consideration),
                                np.array(model_params)))

    return forecasts[::-1]


#### Helper Functions ####


def naming(model_group, ar_order, window_size, regressor, i, j):
    return f"(MG{model_group}, AR{ar_order}, W{window_size}, Regressor = {regressor}, i = {i}, j = {j})"


def save_to_csv(forecast_df, directory, k):
    dataframes = []
    forecast_df_cols = [
        column for column in forecast_df.columns if column != ('Date',) and column != 'time_index']

    for i in range(k):

        new_df = forecast_df.copy()

        for column in forecast_df_cols:
            new_df[column] = forecast_df[column]
            for j in range(len(forecast_df[column])):
                new_df.iloc[j][column] = new_df.iloc[j][column][i]

        dataframes.append(new_df)
    df_names = iter([str(i) for i in range(k)])
    for dataframe in dataframes:
        dataframe.to_csv(
            directory + f'inter_forecasts_k{k}_{next(df_names)}_new.csv')
    return None


def create_forecast_df(T_train, functional_sets):
    headers = ['Date']
    for functional_set in functional_sets:
        for model_group, ar_order, window_size, regressor, i, j in functional_set:
            headers.append(naming(model_group, ar_order,
                           window_size, regressor, i, j))
    forecast_df = df(columns=[headers])
    forecast_df['time_index'] = T_train
    forecast_df.set_index(T_train, inplace=True)
    del forecast_df['time_index']
    return forecast_df


####### Adaptive Learning #########


def adaptive_learning(k,
                      data_and_time,
                      csv_directory,
                      functional_sets,
                      ar_orders,
                      window_sizes,
                      p_norm,
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

        L_Global:

        [L_global_specification, v, lambda] -> E.g. ["EN", [100, 1]]

        L_local:

        E.g.    DL = ["DL"]
                DL_Huber = ["DL_Huber", [0.25,0.5]]
                Ensemble = ["Ensemble", 30]

        L_global

    Return:

        - The set of all models produced by AL
        - The testing and training index
        - AL-induced optimal models
        - AL-induced optimal model forecasts

    """

    # Loading
    L_global_specification = L_global[0]
    L_local_specification = L_local[0]

    if L_global_specification == "EN":
        v, Lambda = L_global[1]
        lambda_vector = [Lambda**(v-i) for i in range(0, v)]

    if L_local_specification == 'DL_Huber':
        quantile_C1, quantile_C2 = L_local[1]

    elif L_local_specification == 'Ensemble':
        v0 = L_local[1]
        v1 = v-v0
        lambda_vector = [Lambda**(v1-i) for i in range(0, v1)]
    """
        NOT IN USE CURRENTLY
        elif L_local_specification=='Ensemble_Huber'

    """

    # Logistics: reading in the forecast dataframes.
    # The CSV files respectively contain [y_t|t-1, y_t|t-2, and y_t|t-3].

    Y_t1_given_t = rc(csv_directory + f'Y_t1_given_t.csv',
                      index_col=['time_index'])

    Y_t2_given_t = rc(csv_directory + f'Y_t2_given_t.csv',
                      index_col=['time_index'])

    Y_t3_given_t = rc(csv_directory + f'Y_t3_given_t.csv',
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
    error_p_norms = create_H_tilda_dict(functional_sets)

    # Step 2 (a).
    for t in tqdm(T_train, leave=True, position=0):
        for model_name in functional_sets:

            # Step 2 (a)(i): Obtain the forecasts [y_t|t-3, y_t|t-2 and y_t|t-1].
            forecast_values = [Y_t1_given_t.loc[t-1, model_name],
                               Y_t2_given_t.loc[t-2, model_name],
                               Y_t3_given_t.loc[t-3, model_name],
                               ]

            # Step 2 (a)(ii): Obtain and save the forecasting error vector.
            val_data = [data_and_time[f'SPY_{k}d_returns'][t]]*k

            if L_local_specification == "DL_Huber":

                forecast_error = list(
                    map(abs, np.subtract(forecast_values, val_data)))

                forecast_errors[model_name][t] = forecast_error

            elif L_local_specification == "DL" or L_local_specification == "Ensemble":

                forecast_error = np.subtract(forecast_values, val_data)

                p_norm_t = np.linalg.norm(
                    x=forecast_error, ord=p_norm)**p_norm

                error_p_norms[model_name][t] = p_norm_t

    # Step 3. Implement Adaptive Learning via the designated option.
    # Step 3 (b):
    if L_local_specification == 'DL_Huber':

        # Step 3(b)(i): Define the local loss function
        # See below (1.7.2) for function declaration

        # Step 3(b)(ii): Declare T^{Huber}:
        T_Huber = np.arange(W_max + 45 + P_max,
                            T_max - 22 + 1)

        # Step 3(b)(iii): Iterate over T^{Huber} to find the constants for each component
        # of the error vector.
        C1, C2 = calculate_quantiles(T_Huber,
                                     forecast_errors,
                                     functional_sets,
                                     quantile_C1,
                                     quantile_C2,
                                     k)

        optimal_models = {}
        y_star_dict = {}

    # Logistics
    elif L_local_specification == "DL":
        optimal_models = {}
        y_star_dict = {}

    elif L_local_specification == "Ensemble":
        optimal_ensemble_weights = create_value_dict(functional_sets)
        ensembled_forecasts_all_t = {}
        p_norm_df = create_p_norm_df_Ensemble(
            error_p_norms, v0, v1, functional_sets, T_test, lambda_vector)

    # Step 3(c).
    for t in tqdm(T_test, leave=True, position=0):

        # Step 3 (c)(i): Declare T_tilda (the adaptive learning lookback window).
        T_tilda = np.arange(t - v + 1, t + 1)

        if L_local_specification == "DL":

            # Step 3 (c)(ii): Exponential-Norm learning.
            h_star, y_star = regular_AL(T_tilda=T_tilda,
                                        t=t,
                                        L_local_specification="DL",
                                        functional_sets=functional_sets,
                                        forecast_df=Y_t3_given_t,
                                        error_p_norms=error_p_norms,
                                        forecast_errors=forecast_errors,
                                        lambda_vector=lambda_vector,
                                        k=k)

            # Step 3 (d): Save this h star (best model) and make an associated forecast
            optimal_models.update({t: h_star})
            y_star_dict.update({t: y_star})

        elif L_local_specification == "Ensemble":

            weights, ensembled_forecasts = ensemble_EN_AL(t=t,
                                                          v0=v0,
                                                          functional_sets=functional_sets,
                                                          forecast_df=Y_t3_given_t,
                                                          p_norm_df=p_norm_df)

            # Save the ensemble weights and the ensemble forecasts
            optimal_ensemble_weights.update({t: weights})
            ensembled_forecasts_all_t.update({t: ensembled_forecasts})

        elif L_local_specification == "DL_Huber":

            y_star, h_star = regular_AL(T_tilda=T_tilda,
                                        t=t,
                                        L_local_specification="DL_Huber",
                                        functional_sets=functional_sets,
                                        forecast_df=Y_t3_given_t,
                                        error_p_norms=error_p_norms,
                                        forecast_errors=forecast_errors,
                                        lambda_vector=lambda_vector,
                                        C1_dict=C1,
                                        C2_dict=C2,
                                        k=k)

            # Step 3 (d): Save this h star (best model) and make an associated forecast
            optimal_models.update({t: h_star})
            y_star_dict.update({t: y_star})

    if L_local_specification == 'Ensemble':

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
               L_local_specification,
               functional_sets,
               forecast_df,
               error_p_norms,
               forecast_errors,
               lambda_vector,
               k,
               C1_dict=None,
               C2_dict=None):

    loss_by_model = {}
    for model_name in functional_sets:

        # Step 3 (c)(ii)(A) Collect the array of p_norms
        # over the adaptive learning lookback window.

        # Step 3 (c)(ii)(A) contd: Evaluate the loss over the period T tilda
        # according to the specific loss function used.
        if L_local_specification == "DL":
            p_norms_T_tilda = [error_p_norms[model_name][tau]
                               for tau in T_tilda]
            loss = exponential_learning(p_norms_T_tilda, lambda_vector)

        elif L_local_specification == "DL_Huber":
            # Step 3 (c)(iii)(A): Evaluate the global loss function,
            # making sure constants are plugged in appropriately.
            loss = L_global(
                T_tilda, forecast_errors[model_name], lambda_vector, C1_dict, C2_dict, k)

        loss_by_model.update({model_name: loss})

    # Step 3 (c)(ii)(B): Find the argmin of the loss function for h in H over the period Tilda.
    h_star = min(loss_by_model, key=loss_by_model.get)

    # Step 3 (c)(ii)(C): Save this h star (best model) and make an associated forecast
    # Note: this forecast is y_{t+3|t}
    y_star = forecast_df.loc[t, h_star]

    return y_star, h_star

###### 1.7.1 Exponential-Norm Learning Function ######


def exponential_learning(p_norms_T_tilda, lam):
    return np.dot(p_norms_T_tilda, lam)

###### 1.7.2 Learning Function with Huber Loss ######

# Step 3 (c)(iii)(1): Define the local loss function and the Scalar Huber Function.


def local_Huber(x, C1, C2, k):
    local_loss = 0
    for i in range(k):
        local_loss += Scalar_Huber(x[i], C1[i], C2[i])
    return local_loss


def Scalar_Huber(x_i, C1_i, C2_i):
    if x_i >= C2_i:
        return (C2_i - C1_i)*(x_i) + (C1_i**2 - C2_i**2)/2
    elif (C2_i > x_i and x_i > C1_i):
        return ((x_i - C1_i)**2)/2
    else:
        return 0

# Step 3 (c)(iii)(2): Calculate the quantiles of the error distributions.


def calculate_quantiles(T_Huber, forecast_errors, functional_sets, quantile_C1, quantile_C2, k):
    C1 = {}
    C2 = {}
    for tau in T_Huber:
        # Step 3 (c)(iii)(2)(a): Collect the error vector for each model in the functional sets.
        quantile_set_C1 = []
        quantile_set_C2 = []
        for i in range(k):
            E = [forecast_errors[model][tau][i] for model in functional_sets]
            quantile_set_C1.append(np.quantile(E, quantile_C1))
            quantile_set_C2.append(np.quantile(E, quantile_C2))
        C1.update({tau: quantile_set_C1})
        C2.update({tau: quantile_set_C2})
    return C1, C2

# Step 3 (c)(iii)(3): Calculate the global loss.


def L_global(T_tilda, forecast_errors, lambda_vector, C1_dict, C2_dict, k):
    global_loss = []
    for tau in T_tilda:
        x = forecast_errors[tau]
        global_loss.append(local_Huber(x, C1_dict[tau], C2_dict[tau], k))
    return np.dot(global_loss, lambda_vector)

###### 1.7.3 Ensemble ######


def ensemble_EN_AL(t, v0, functional_sets, forecast_df, p_norm_df):

    # Step 1: Declare T_0.
    T_0 = np.arange(t - v0 + 1, t+1)

    minimising_model_count = create_value_dict(functional_sets)

    # Re-write starts here
    # Step 2.
    for s in T_0:

        model_with_min_loss = p_norm_df.loc[s].idxmin(axis=1)

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

    return weights, ensembled_forecasts


def create_p_norm_df_Ensemble(error_p_norms, v0, v1, functional_sets, T_test, lambda_vector):
    p_norm_df = df(np.arange(T_test[0] - v0 + 1,
                             T_test[-1] + 1), columns=['s'])
    p_norm_df.set_index(p_norm_df['s'], inplace=True)

    del p_norm_df['s']
    for s in tqdm(p_norm_df.index, leave=True, position=0):
        T_1 = np.arange(s - v1 + 1, s + 1)
        for model in functional_sets:
            error_list = [error_p_norms[model][tau]
                          for tau in T_1]
            p_norm_df.loc[s, model] = exponential_learning(
                error_list, lambda_vector)

    return p_norm_df

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
