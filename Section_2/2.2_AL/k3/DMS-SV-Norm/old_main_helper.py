from pandas import DataFrame as df
###### Helper Functions for implementation ######


def save_models_and_forecasts(AL_output,
                              learning_specifications,
                              k,
                              functional_sets,
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

    if LS == 'DMS-SV-Norm':
        # optimal_model_df = df(columns = ['h_star', 'y_star'])
        optimal_model_df['h_star'] = df(
            AL_output[2].values()).set_index(T_test)
        optimal_model_df['y_star'] = df(
            AL_output[3].values()).set_index(T_test)
        optimal_model_df.to_csv(
            f'../../../../../../DataCentre/RSS_AL/k{k}/Learning_Results/{LS}/Loss_{LS}_p_{learning_specifications[1][1]}_Lambda_{learning_specifications[1][2]}_k_{k}.csv')

    if LS == 'Ensemble-SV-Norm':
        optimal_model_df['Ensembled Forecast'] = df(
            AL_output[3].values()).set_index(T_test)
        optimal_model_df.to_csv(f'../../../../../../DataCentre/RSS_AL/k{k}/Learning_Results/{LS}/'
                                + f'Loss_{LS}_p_{learning_specifications[1][1]}_Lambda_{learning_specifications[1][2]}_v0_{learning_specifications[2][0]}_k_{k}.csv')

    return None
