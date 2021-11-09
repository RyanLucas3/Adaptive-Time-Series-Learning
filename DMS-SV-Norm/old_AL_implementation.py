import old_AL
import old_main_helper as helper
import sys


# Modelling Specifications
k = int(sys.argv[1])
AL_specification = "EN"

ar_orders = list(range(1, 11))
window_sizes = [22, 63, 126, 252]

data = old_AL.rc(
    '../../../../../../DataCentre/jun_data/merged_data_for_paper.csv')

desired_model_groups = ["MG1", "MG2N", "MG2T",
                        "MG3N", "MG3T", "MG2V", "MG3V", "MG4V"]

functional_sets = old_AL.ModelGroupSpecs(data_and_time=data,
                                         ar_orders=ar_orders,
                                         window_sizes=window_sizes,
                                         desired_model_groups=desired_model_groups).generate_H_tilda()
v = 100
# Lambda = 0.9
# p = 1
v0 = 25

for p in [1]:
    for Lambda in [1]:

        if AL_specification == "EN":

            specification_learning = ['EN', [v, p, Lambda]]

            adaptive_learning = old_AL.adaptive_learning(k,
                                                         data_and_time=data,
                                                         csv_directory=f'../../../../../../DataCentre/AdaptiveLearning/Real_data/Forecasts/VAR_2x/Y_t3_given_t.csv',
                                                         functional_sets=functional_sets,
                                                         ar_orders=ar_orders,
                                                         window_sizes=window_sizes,
                                                         specification_learning=specification_learning
                                                         )

        elif AL_specification == "Ensemble_EN":

            specification_learning = ['Ensemble_EN', [v, p, Lambda], [v0]]

            adaptive_learning = old_AL.adaptive_learning(k,
                                                         data_and_time=data,
                                                         csv_directory=f'../../../../../../DataCentre/AdaptiveLearning/Real_data/Forecasts/VAR_2x/Y_t3_given_t.csv',
                                                         functional_sets=functional_sets,
                                                         ar_orders=ar_orders,
                                                         window_sizes=window_sizes,
                                                         specification_learning=specification_learning
                                                         )

        if AL_specification == "EN":
            specification_learning = ['DMS-SV-Norm', [v, p, Lambda]]

        elif AL_specification == "Ensemble_EN":
            specification_learning = ['Ensemble-SV-Norm', [v, p, Lambda]]

        helper.save_models_and_forecasts(
            adaptive_learning, specification_learning, k, desired_model_groups, False)
