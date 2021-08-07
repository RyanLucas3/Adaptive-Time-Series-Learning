from pandas import DataFrame as df
###### Helper Functions for implementation ######


def save_models_and_forecasts(AL_output,
                              learning_specifications,
                              k,
                              functional_sets,
                              model_groups,
                              saveweight=False,
                              ):
    optimal_model_df = df()
    T_test = AL_output[1][0]
    optimal_model_df['time_index'] = T_test
    optimal_model_df.set_index(T_test, inplace=True)

    if saveweight:
        for model in functional_sets:
            optimal_model_df[f'{model} weight'] = 0
            for t in T_test:
                optimal_model_df.loc[t,
                                     f'{model} weight'] = AL_output[2][t][model]

    LS = learning_specifications[0]

    if LS == 'EN':
        # optimal_model_df = df(columns = ['h_star', 'y_star'])
        optimal_model_df['h_star'] = df(
            AL_output[2].values()).set_index(T_test)
        optimal_model_df['y_star'] = df(
            AL_output[3].values()).set_index(T_test)
        optimal_model_df.to_csv(
            f'../../DataCentre/AdaptiveLearning/Real_data/Learning_Results/Inter_forecasts/{LS}/Loss_{LS}_p_{learning_specifications[1][1]}_Lambda_{learning_specifications[1][2]}_k_{k}.csv')

    if LS == 'Ensemble_EN':
        optimal_model_df['Ensembled Forecast'] = df(
            AL_output[3].values()).set_index(T_test)
        optimal_model_df.to_csv(f'../../DataCentre/AdaptiveLearning/Real_data/Learning_Results/Inter_forecasts/{LS}/'
                                + f'Loss_{LS}_p_{learning_specifications[1][1]}_Lambda_{learning_specifications[1][2]}_v0_{learning_specifications[2]}_k_{k}_2020.csv')

    elif LS == 'Huber':
        optimal_model_df['h_star'] = df(
            AL_output[2].values()).set_index(T_test)
        optimal_model_df['y_star'] = df(
            AL_output[3].values()).set_index(T_test)
        optimal_model_df.to_csv(f'../../DataCentre/AdaptiveLearning/Real_data/Learning_Results/Inter_forecasts/{LS}/'
                                + f'Loss_{LS}_p_{learning_specifications[1][1]}_Lambda_{learning_specifications[1][2]}_C_{learning_specifications[2]}_k_{k}.csv')

    return None


def merge_adaptive_learning_dataframe(T_test):
    merged_forecasts = df(T_test)
    merged_models = df(T_test)
    for k in [3, 5, 10, 15, 22]:
        for v in [50, 100]:
            for p in [1, 2]:
                for Lambda in [0.8, 0.85, 0.9, 0.95, 0.99, 1]:
                    h_star = rc(
                        f'../../DataCentre/AdaptiveLearning/Real_data/Learning_Results/v_{v}_p_{p}_Lambda_{Lambda}_k_{k}.csv')['h_star']
                    y_star = rc(
                        f'../../DataCentre/AdaptiveLearning/Real_data/Learning_Results/v_{v}_p_{p}_Lambda_{Lambda}_k_{k}.csv')['h_star']

                    merged_forecasts[f'forecasts_v_{v}_p_{p}_Lambda_{Lambda}_k_{k}'] = y_star
                    merged_models[f'forecasts_v_{v}_p_{p}_Lambda_{Lambda}_k_{k}'] = h_star
    merged_forecasts.to_csv(
        '../../DataCentre/AdaptiveLearning/Real_data/Learning_Results/merged_AL_forecasts.csv')
    merged_models.to_csv(
        '../../DataCentre/AdaptiveLearning/Real_data/Learning_Results/merged_AL_models.csv')
    return None
