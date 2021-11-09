import main_helper as helper
import MC_General
import sys

# Modelling Specifications
k = int(sys.argv[1])

AL_specification = "Ensemble"

ar_orders = list(range(1, 11))

method = "MG2N"

folder = "MCG"

model_g = '(MG2N, ARNone, W252, Regressor = VIX_slope, i = None, j = None)'

window_sizes = [22, 63, 126, 252]

data = MC_General.rc(
    f'../../../../../../DataCentre/jun_data/merged_data_for_paper.csv')

desired_model_groups = ["MG1", "MG2N", "MG2T",
                        "MG3N", "MG3T", "MG2V", "MG3V", "MG4V"]

functional_sets = MC_General.ModelGroupSpecs(data_and_time=data,
                                             ar_orders=ar_orders,
                                             window_sizes=window_sizes,
                                             desired_model_groups=desired_model_groups).generate_H_tilda()

v = 100
v0 = 50

for Lambda in [0.98, 0.99, 1]:
    for p_norm in [1, 2]:

        L_global_specification = ["EN", [v, Lambda]]

        if AL_specification == "DL":

            L_local_specification = ['DL']

        elif AL_specification == "Ensemble":

            L_local_specification = ['Ensemble', v0]

        adaptive_learning = MC_General.adaptive_learning(k,
                                                         data_and_time=data,
                                                         csv_directory=f'../../../../../../DataCentre/AdaptiveLearning/Real_data/Forecasts/VAR_2x/',
                                                         functional_sets=functional_sets,
                                                         ar_orders=ar_orders,
                                                         window_sizes=window_sizes,
                                                         p_norm=p_norm,
                                                         L_local=L_local_specification,
                                                         L_global=L_global_specification,
                                                         model_g=model_g
                                                         )

        if AL_specification == "DL":
            L_local_specification = ['DMS-MV-MC']
        elif AL_specification == "Ensemble":
            L_local_specification = ['Ensemble-MV-MC', v0]

        helper.save_models_and_forecasts(
            adaptive_learning, L_global_specification, L_local_specification, k, functional_sets, p_norm, False, method)
