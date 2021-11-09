import train_and_forecast
import main_helper as helper
import sys

# Modelling Specifications
k = int(sys.argv[1])

AL_specification = "Ensemble"

ar_orders = list(range(1, 11))

method = "Test"

window_sizes = [22, 63, 126, 252]

data = train_and_forecast.rc(
    f'../../../../../../DataCentre/jun_data/merged_data_for_paper.csv')

desired_model_groups = ["MG1", "MG2N", "MG2T",
                        "MG3N", "MG3T", "MG2V", "MG3V", "MG4V"]

functional_sets = train_and_forecast.ModelGroupSpecs(data_and_time=data,
                                                     ar_orders=ar_orders,
                                                     window_sizes=window_sizes,
                                                     desired_model_groups=desired_model_groups).generate_H_tilda()


v = 100
v0 = 75

for Lambda in [0.99]:
    for p_norm in [2]:

        L_global_specification = ["EN", [v, Lambda]]

        if AL_specification == "DL":

            L_local_specification = ['DL']

        elif AL_specification == "Ensemble":

            L_local_specification = ['Ensemble', v0]

        adaptive_learning = train_and_forecast.adaptive_learning(k,
                                                                 data_and_time=data,
                                                                 csv_directory=f'../../../../../../DataCentre/RSS_AL/k{k}/Forecasts/',
                                                                 functional_sets=functional_sets,
                                                                 ar_orders=ar_orders,
                                                                 window_sizes=window_sizes,
                                                                 p_norm=p_norm,
                                                                 L_local=L_local_specification,
                                                                 L_global=L_global_specification,
                                                                 )

        if AL_specification == "DL":
            L_local_specification = ['DMS-MV-Norm']

        elif AL_specification == "Ensemble":
            L_local_specification = ['Ensemble-MV-Norm', v0]

        helper.save_models_and_forecasts(
            adaptive_learning, L_global_specification, L_local_specification, k, functional_sets, p_norm, False, method)
