from os import error
import numpy as np
from tqdm import tqdm
from pandas import DataFrame as df
from pandas import read_csv as rc
from pandas import DatetimeIndex as dti
import statistics as stats
import pandas as pd
import json


def statistical_evaluation(
        k,
        T_model,
        data,
        fixed_model_forecasts,
        AL_forecasts
):
    """
    Input specifications (by example):

    forecast_horizon=3, known as $k$ in our main work. 
        Choices include 5, 10, 15, 22.

    T_model. This is the index for which we have forecasts.

    data. This should be a pandas dataframe including (1) a time index, (2) the k-step ahead SPY returns
    and (3) the SPY close prices. 

    N.B.    The k-step ahead forecast, usually noted as $y_{t+k|t}$ is indexed at time t. 
            See step 1 for further insights.

    fixed_model_forecats. 

    AL_forecasts. These are forecasts produced by adaptive learning models.

    Return:

    - The number of forecasts made.
    - MAE for all models.
    - MSE for all models.
    - The percentage of the time that the model made the directionally correct prediction (+-)

    """

    # Define H_bar as the combined set of fixed models and adaptive learning models.
    H_bar_forecast_df = AL_forecasts.merge(
        fixed_model_forecasts, left_index=True, right_index=True)
    all_models = [model for model in H_bar_forecast_df.columns if model !=
                  'Date' and model != 'time_index']
    # Logistics
    error_dict = create_error_dict(all_models)
    direction_correct_dict = create_value_dict(all_models)

    # Statistical Evaluation: Step 1
    for t in tqdm(T_model, leave=True, position=0):

        # Step 1:
        for model in all_models:

            # Access the forecasts and the validation data
            forecast_values = H_bar_forecast_df.loc[t, model]
            val_data = data[f'SPY_{k}d_returns'][t+k]

            # Obtain and save the forecasting error.
            errors = forecast_values - val_data
            error_dict[model][str(t+k)] = errors

            # Obtain whether the forecast was directionally correct
            if (forecast_values > 0 and val_data > 0) or (forecast_values < 0 and val_data < 0):
                direction_correct_dict[model] += 1

    # Logistics
    MAE_dict = create_value_dict(all_models)
    MSE_dict = create_value_dict(all_models)
    perc_correct_dict = create_value_dict(all_models)

    # Statistical Evaluation: Step 2.
    for model in tqdm(all_models, leave=True, position=0):
        MAE_dict[model] = get_MAE(error_dict, model, T_model)
        MSE_dict[model] = get_MSE(error_dict, model, T_model)
        perc_correct_dict[model] = get_perc_correct(
            direction_correct_dict, model, T_model)

    number_of_forecasts = len(T_model)

    # Get metrics for long only S&P.
    MAE_dict["Long Only"] = np.nan
    MSE_dict["Long Only"] = np.nan
    perc_correct_dict["Long Only"] = 100*(1/len(T_model))*sum(
        [1 for t in T_model if data.loc[t, f'SPY_{k}d_returns'] > 0])

    with open(f'/users/ryanlucas/Desktop/OPRG1/DataCentre/AdaptiveLearning/error_dict_k{k}.json', 'w') as file:
        json.dump(error_dict, file)
    # Return the output.
    return number_of_forecasts, MAE_dict, MSE_dict, perc_correct_dict


def trading_evaluation(
        k,
        T_pi,
        data,
        fixed_model_forecasts,
        AL_forecasts
):
    """
    Input specifications (by example):

    forecast_horizon = 3, known as $k$ in our main work. 
    Choices include 5, 10, 15, 22.

    T_pi. The index for profit calculation.

    data. This should be a pandas dataframe including 
    (1) a time index, (2) the k-step ahead SPY returns and (3) the SPY close prices.

    fixed_model_forecats. These are forecasts made by the trained fixed models.

    AL_forecasts. These are forecasts produced by adaptive learning models.

    Return:

    - The number of forecasts made.
    - MAE for all models.
    - MSE for all models.
    - The percentage of the time that the model made the directionally correct prediction (+-).

    """
    H_bar_forecast_df = AL_forecasts.merge(
        fixed_model_forecasts, left_index=True, right_index=True)
    all_models = [model for model in H_bar_forecast_df.columns if model !=
                  'Date' and model != 'time_index']
    all_models.append("Long Only")
    H_bar_forecast_df["Long Only"] = 1
    # Trading Evaluation: Step 1:
    # Logistics
    signals = create_value_dict(all_models)
    daily_profit_df = create_model_df(T_pi, data, all_models)
    cumulative_profit_df = create_model_df(T_pi, data, all_models)

    for t in tqdm(T_pi, leave=True, position=0):
        daily_profit_t = []
        date = data["Date"].loc[t]
        for model in all_models:

            # Produce a trading signal and calculate daily profit.
            daily_profit_t.append(trading_strategy(model_forecasts=H_bar_forecast_df[model],
                                                   SPY_close=data['spy_close'],
                                                   k=k,
                                                   t=t))

        daily_profit_df.loc[date, :] = daily_profit_t

    # Trading Evaluation: Step 2:
    annualised_return = create_value_dict(all_models)
    sharpe_ratio = create_value_dict(all_models)
    max_drawdowns = create_value_dict(all_models)
    max_drawdown_dates = {}

    # Calculate trading metrics.
    for model in tqdm(all_models, leave=True, position=0):

        cumulative_profit_df[model] = df(np.cumsum(daily_profit_df[model]))
        annualised_return[model] = get_annualised_return(
            cumulative_profit_df, model, T_pi)

        sharpe_ratio[model] = get_sharpe_ratio(daily_profit_df, model)
        max_drawdowns[model] = get_max_dd_and_date(
            cumulative_profit_df, model)[0]
        max_drawdown_dates.update(
            {model: get_max_dd_and_date(cumulative_profit_df, model)[1]})

    # Return the output.
    return cumulative_profit_df, annualised_return, sharpe_ratio, max_drawdowns, max_drawdown_dates

#### Statistical Metrics Calculations ####


def get_MAE(error_dict, model, T_model):
    abs_errors = map(abs, list(error_dict[model].values()))
    sum_abs_errors = sum(abs_errors)
    return (1/len(T_model))*sum_abs_errors


def get_MSE(error_dict, model, T_model):
    errors_as_array = np.array(list(error_dict[model].values()))
    sum_squared_errors = sum(np.power(errors_as_array, 2))
    return (1/len(T_model))*sum_squared_errors


def get_perc_correct(direction_correct_dict, model, T_model):
    amount_correct = direction_correct_dict[model]
    return 100*(1/len(T_model))*amount_correct

#### Trading Metrics Calculations ####


def trading_strategy(model_forecasts, SPY_close, k, t):

    PL_t = 0
    signal_t_minus_1 = 0

    # Long if return prediction > 0 ; otherwise short.
    for i in range(1, k+1):
        if model_forecasts[t-i] > 0:
            signal_t_minus_1 += 1
        elif model_forecasts[t-i] < 0:
            signal_t_minus_1 -= 1

    PL_t += (1/k)*signal_t_minus_1 * \
        ((SPY_close[t] - SPY_close[t-1])/SPY_close[t-1])

    return float(PL_t)


def get_sharpe_ratio(daily_profit_df, model):
    mean = daily_profit_df[model].mean()
    std_dev = daily_profit_df[model].std()
    return 252**(0.5)*mean/std_dev


def get_max_dd_and_date(cumulative_profit_df, model):
    rolling_max = (cumulative_profit_df[model]+1).cummax()
    period_drawdown = (
        ((1+cumulative_profit_df[model])/rolling_max) - 1).astype(float)
    drawdown = round(period_drawdown.min(), 3)
    drawdown_date = str(cumulative_profit_df[model][: period_drawdown.idxmin()].astype(
        float).idxmax())[0:10] + '  --  ' + str(period_drawdown.astype(float).idxmin())[0:10]
    return drawdown, drawdown_date


def get_annualised_return(cumulative_profit_df, model, T_pi):
    return cumulative_profit_df[model].iloc[-1]*(252/len(T_pi))

#### Helper Functions ####


def create_H_bar_dict(all_models):
    H_bar = {}
    for model in all_models:
        H_bar.update({model: {}})
    return H_bar


def create_value_dict(all_models):
    df_dict = {}
    for model in all_models:
        df_dict.update({model: 0})
    return df_dict


def create_model_df(index, data, all_models):
    headers = []
    for model in all_models:
        headers.append(model)
    model_df = df(columns=headers)
    model_df['date_index'] = data['Date'].loc[index]
    model_df.set_index(dti(data['Date'].loc[index]), inplace=True)
    del model_df['date_index']
    return model_df


def create_evaluation_df(k, period, statistical_evaluation, trading_evaluation):
    headers = ['Model Name', 'MAE', 'MSE', 'Percentage Correct',
               'Annualised Return', 'Sharpe Ratio', 'MDD', 'MDD Date']
    values = [statistical_evaluation[1].keys(), statistical_evaluation[1].values(), statistical_evaluation[2].values(), statistical_evaluation[3].values(
    ), trading_evaluation[1].values(), trading_evaluation[2].values(), trading_evaluation[3].values(), trading_evaluation[4].values()]
    eval_df = df(columns=headers)
    for i in range(len(headers)):
        eval_df[headers[i]] = values[i]
    eval_df.to_csv(
        f'../../DataCentre/AdaptiveLearning/Real_data/Evaluation/Evaluation_{period}_VAR_k{k}_.csv')
    return None


def create_error_dict(all_models):
    df_dict = {}
    for model in all_models:
        df_dict.update({model: {}})
    return df_dict


def export_cumulative_profit_df(cumulative_profit_df, k, period):
    cumulative_profit_df.to_csv(
        f'../../DataCentre/AdaptiveLearning/Real_data/Evaluation/cumulative_profit_{period}_VAR_k{k}_.csv')
