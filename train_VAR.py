import numpy as np
from itertools import product
from pandas import DataFrame as df
from pandas import read_csv as rc
from tqdm import tqdm
from statsmodels.tsa.api import VAR


class ModelGroupSpecs:
    def __init__(self, data_and_time, ar_orders, window_sizes, desired_model_groups):
        self.ar_orders = ar_orders
        self.window_sizes = window_sizes
        self.data_and_time = data_and_time
        self.desired_model_groups = desired_model_groups

    def get_all_possible_combinations(self, model_group, MG_ar_orders, MG_window_sizes, MG_regressors, MG_i=[None], MG_j=[None]):
        models = list(product(model_group, MG_ar_orders,
                      MG_window_sizes, MG_regressors, MG_i, MG_j))
        screened_models = []
        for model in models:
            if (model[0] == '3V' or model[0] == '4V'):
                if (12 + 9*model[1]) < model[2]:
                    screened_models.append(model)
            elif model[0] == '2V':
                if (6 + 4*model[1]) < model[2]:
                    screened_models.append(model)
        return screened_models

    def create_functional_sets(self):
        output = []

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
            self.MG3V_model_specs_VIX = self.get_all_possible_combinations(model_group=['3V'],
                                                                           MG_ar_orders=[
                1, 2, 3, 4, 5],
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

                if model_group == '2V':

                    data = np.column_stack([data_and_time[f'SPY_{k}d_returns'],
                                            data_and_time[regressor]])

                elif model_group == "3V":

                    data = np.column_stack([data_and_time[f'SPY_{k}d_returns'],
                                            data_and_time[i],
                                            data_and_time[j]])

                elif model_group == "4V":

                    data = np.column_stack([data_and_time[f'SPY_{k}d_returns'],
                                            data_and_time[i],
                                            data_and_time[j]])

                windowed_data = window_data(data,
                                            t,
                                            window_size)

                model = VAR(windowed_data).fit(ar_order)

                forecasts = []

                for i in range(k):
                    forecasts.append(model.forecast(
                        y=data[t-(ar_order)+i+1:t+1+i, :], steps=k-i)[-1][0])

                # forecast = model.forecast(
                #     y=windowed_data[-(ar_order):, :], steps=k)[-1][0]

                forecasts_t.append(forecasts)

        forecast_df.loc[t] = forecasts_t

    return forecast_df


def save_to_csv(forecast_df, directory):
    forecast_df.to_csv(directory)
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


def window_data(data, t, w):
    return np.array(data[(t - w + 1): (t + 1)])

##### Individual Training Functions See Appendix 1.5. of the pseudo-algorithm #####

################################################################################

#### Individual Forecasting Functions. See Appendix 1.6 of the Psuedo-Algo. ####

#### Helper Functions ####


def naming(model_group, ar_order, window_size, regressor, i, j):
    return f"(MG{model_group}, AR{ar_order}, W{window_size}, Regressor = {regressor}, i = {i}, j = {j})"
