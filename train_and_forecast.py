import numpy as np
import pandas as pd
from itertools import product
from pandas import DataFrame as df
from pandas import read_csv as rc
from statsmodels.tsa.vector_ar.var_model import forecast
from tqdm import tqdm
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA


class ModelGroupSpecs:
    def __init__(self, data_and_time, ar_orders, window_sizes, desired_model_groups):
        self.ar_orders = ar_orders
        self.window_sizes = window_sizes
        self.data_and_time = data_and_time
        self.desired_model_groups = desired_model_groups

    def get_all_possible_combinations(self, model_group, MG_ar_orders, MG_window_sizes, MG_regressors, MG_i=[None], MG_j=[None]):
        return list(product(model_group, MG_ar_orders, MG_window_sizes, MG_regressors, MG_i, MG_j))

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
        Choices include 5, 10, 15, 22.

        data_and_time. This should be a pandas dataframe including (1) a time index, (2) the dependent variable of interest
        and (3) exogenous variables.

        functional_sets. This is the set of all models to be trained. It is obtained via the ModelGroupSpecs class.

        ar_orders = [1, 2,..., 10] 

        window_sizes = [22, 63, 126, 252].

    Return:

        - The set of forecasts produced by all fixed model groups.

    """

    # Step 1 (a): Defining our indexes for training and testing.
    T_max = len(data_and_time) - 1
    W_max = max(window_sizes)
    P_max = max(ar_orders)
    T_train = np.arange(W_max + 22 + P_max, T_max - 22 + 1)
    # Logistics
    # Creation of H_tilda

    forecast_df = create_forecast_df(T_train, functional_sets)

    # Step 2 (a): Training the models and making a forecast.
    for t in tqdm(T_train, leave=True, position=0):

        forecasts_t = [data_and_time['Date'][t]]

        for functional_set in functional_sets:
            for model_group, ar_order, window_size, regressor, i, j in functional_set:

                # model_name = naming(model_group, ar_order, window_size, regressor, i, j)

                dep_train_data = window_dep_variable(data_and_time[f'SPY_{k}d_returns'],
                                                     t,
                                                     window_size)

                if model_group == '1':

                    model = ARIMA(dep_train_data,
                                  order=(ar_order, 0, 0)).fit(method="yule_walker")

                    forecasts = forecast_MG1(
                        model_params=model.params,
                        dep=data_and_time[f'SPY_{k}d_returns'],
                        ar_order=ar_order,
                        k=k,
                        t=t)

                    forecasts_t.append(forecasts)

                elif model_group == '2N':

                    model = train_MG2N(data_and_time[regressor],
                                       dep_train_data,
                                       t,
                                       window_size,
                                       k)

                    forecasts = [model.params[0] + model.params[1]
                                 * data_and_time[regressor][t]]*3

                    forecasts_t.append(forecasts)

                elif model_group == '3N':

                    high_low_point = np.column_stack([data_and_time[i],
                                                      data_and_time[j]])

                    model = train_MG3N(high_low_point,
                                       dep_train_data,
                                       t,
                                       window_size,
                                       k)

                    forecasts = [
                        np.dot(model.params, sm.add_constant(high_low_point)[t])]*3

                    forecasts_t.append(forecasts)

                elif model_group == '2T':

                    model = train_MG2T(
                        dep_train_data=dep_train_data,
                        dep_to_be_lagged=data_and_time[f'SPY_{k}d_returns'],
                        regressor=data_and_time[regressor],
                        t=t,
                        w=window_size,
                        ar_order=ar_order,
                        k=k)

                    forecasts_t.append(forecast_with_lags(
                        model_params=model.params,
                        dep_to_be_lagged=data_and_time[f'SPY_{k}d_returns'],
                        regressor=data_and_time[regressor],
                        t=t,
                        ar_order=ar_order,
                        k=k))

                if model_group == '3T':

                    high_low_point = np.column_stack([data_and_time[i],
                                                      data_and_time[j]])

                    model = train_MG3T(dep_train_data=dep_train_data,
                                       dep_to_be_lagged=data_and_time[f'SPY_{k}d_returns'],
                                       regressor=high_low_point,
                                       t=t,
                                       w=window_size,
                                       ar_order=ar_order,
                                       k=k)

                    forecasts_t.append(forecast_with_lags(
                        model_params=model.params,
                        dep_to_be_lagged=data_and_time[f'SPY_{k}d_returns'],
                        regressor=high_low_point,
                        t=t,
                        ar_order=ar_order,
                        k=k))

        forecast_df.loc[t] = forecasts_t

    return forecast_df


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
            directory + f'inter_forecasts_k{k}_{next(df_names)}.csv')
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


def window_dep_variable(dep_variable, t, w):
    return np.array(dep_variable[(t - w + 1): (t + 1)])


def window_exog_variables(exog_variables, t, w, k):
    return np.array(exog_variables[(t - w - k + 1): (t - k + 1)])

##### Individual Training Functions See Appendix 1.5. of the pseudo-algorithm #####


def train_MG2N(regressor, dep_train_data, t, w, k):

    indep_train_data = window_exog_variables(regressor, t, w, k)
    indep_train_data = sm.add_constant(indep_train_data)

    model = sm.OLS(dep_train_data,
                   indep_train_data)

    model = model.fit()

    return model


def train_MG3N(regressor, dep_train_data, t, w, k):

    indep_train_data = window_exog_variables(regressor, t, w, k)
    indep_train_data = sm.add_constant(indep_train_data)

    model = sm.OLS(dep_train_data,
                   indep_train_data)

    model = model.fit()

    return model


def train_MG2T(dep_train_data, dep_to_be_lagged, regressor, t, w, ar_order, k):

    indep_train_data = vectorise_indep_variables(dep_to_be_lagged,
                                                 exog=regressor,
                                                 t=t,
                                                 ar_order=ar_order,
                                                 w=w,
                                                 k=k)

    model = sm.OLS(dep_train_data, sm.add_constant(indep_train_data))

    model = model.fit()

    return model


def train_MG3T(dep_train_data, dep_to_be_lagged, regressor, t, w, ar_order, k):

    indep_train_data = vectorise_indep_variables(dep_to_be_lagged,
                                                 regressor,
                                                 t,
                                                 ar_order,
                                                 w,
                                                 k)

    model = sm.OLS(dep_train_data, sm.add_constant(indep_train_data))

    model = model.fit()

    return model


def vectorise_indep_variables(dep_to_be_lagged, exog, t, ar_order, w, k):

    lagged_dep_data = list(dep_to_be_lagged[(t-w-ar_order): t])
    lagged_dep_data = lagged_dep_data[::-1]

    lagged_p = [lagged_dep_data[i: i+ar_order]
                for i in range(0, len(lagged_dep_data) - ar_order)]
    lagged_p = np.array(lagged_p[::-1])

    exog = window_exog_variables(exog, t, w, k)

    vectorised_training_data = np.array(
        [np.append(lagged_p[i], [exog[i]]) for i in range(0, len(exog))])

    return vectorised_training_data

################################################################################

#### Individual Forecasting Functions. See Appendix 1.6 of the Psuedo-Algo. ####


def forecast_MG1(model_params, dep, ar_order, k, t):

    forecasts_k = []

    coefficient_vector = np.array(model_params[1:-1])

    C = model_params[0]*(1-sum(coefficient_vector))

    for i in range(0, k):

        inter_forecasts = []

        previous_observations_y = list(dep[(t-ar_order+1+i): t+i+1])

        for tau in range(0, k-i):  # tau = 0 means 1-step-ahead.

            observations_under_consideration = np.array(
                inter_forecasts[::-1][:ar_order] + previous_observations_y[tau:][::-1])

            inter_forecasts.append(
                np.dot(observations_under_consideration, coefficient_vector) + C)

        forecasts_k.append(inter_forecasts[-1])

    return forecasts_k


def forecast_with_lags(model_params, dep_to_be_lagged, regressor, t, ar_order, k):

    forecasts = []

    for i in range(0, k):

        inter_forecasts = []

        lagged_y = np.array(
            dep_to_be_lagged[(t - ar_order + i + 1): (t + i + 1)])
        lagged_x = np.array(regressor[(t - k + i + 1): (t + 1)])

        for tau in range(0, k-i):  # tau = 0 means 1-step-ahead.

            observations_under_consideration = [[1]]

            if len(inter_forecasts) > 0:
                observations_under_consideration.append(
                    inter_forecasts[::-1][:ar_order])

            observations_under_consideration.append(lagged_y[tau:][::-1])

            observations_under_consideration.append(lagged_x[tau].flatten())

            observations_under_consideration = np.concatenate(
                observations_under_consideration).ravel()

            inter_forecasts.append(np.dot(np.array(observations_under_consideration),
                                          np.array(model_params)))

        forecasts.append(inter_forecasts[-1])

    return forecasts

#### Helper Functions ####


def naming(model_group, ar_order, window_size, regressor, i, j):
    return f"(MG{model_group}, AR{ar_order}, W{window_size}, Regressor = {regressor}, i = {i}, j = {j})"
