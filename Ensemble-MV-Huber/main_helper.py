from pandas import DataFrame as df

###### Helper Functions for implementation ######


def save_models_and_forecasts(AL_output,
                              L_global_specification,
                              L_local_specification,
                              k,
                              functional_sets,
                              p_norm,
                              saveweight=False,
                              method="M0",

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

    LS = L_local_specification[0]

    if LS == 'Ensemble-MV-Huber':
        optimal_model_df['Ensembled Forecast'] = df(
            AL_output[3].values()).set_index(T_test)
        optimal_model_df.to_csv(f'../../../../../../DataCentre/RSS_AL/k{k}/Learning_Results/{LS}/'
                                + f'Loss_{LS}_p_{p_norm}_Lambda_{L_global_specification[1]}_v0_{L_local_specification[1]}_k_{k}_{method}.csv')

    elif LS == 'DMS-MV-Huber':
        optimal_model_df['h_star'] = df(
            AL_output[2].values()).set_index(T_test)
        optimal_model_df['y_star'] = df(
            AL_output[3].values()).set_index(T_test)
        optimal_model_df.to_csv(f'../../../../../../DataCentre/RSS_AL/k{k}/Learning_Results/{LS}/'
                                + f'Loss_{LS}_p_{p_norm}_Lambda_{L_global_specification[1]}_C_{L_local_specification[1]}_k_{k}_{method}.csv')

    return None
