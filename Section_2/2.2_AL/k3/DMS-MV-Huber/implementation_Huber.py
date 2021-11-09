import Huber_AL
import main_helper as helper
import sys

# Modelling Specifications
k = int(sys.argv[1])

AL_specification = "Huber-Norm"

ar_orders = list(range(1, 11))

# Method could for instance be the variant of AL you want to run
# Or alternatively if you want to just save the results with a different name.

method = "Huber-Norm"

window_sizes = [22, 63, 126, 252]

data = Huber_AL.rc(
    f'../../../../../../DataCentre/jun_data/merged_data_for_paper.csv')

desired_model_groups = ["MG1", "MG2N", "MG2T",
                        "MG3N", "MG3T", "MG2V", "MG3V", "MG4V"]

functional_sets = Huber_AL.ModelGroupSpecs(data_and_time=data,
                                           ar_orders=ar_orders,
                                           window_sizes=window_sizes,
                                           desired_model_groups=desired_model_groups).generate_H_tilda()


v = 100
v0 = 25
C = 0.7
p = 1
for C in [0.25]:
    for v0 in [75]:
        for Lambda in [0.96]:

            L_global_specification = [v, Lambda]

            if AL_specification == "Huber-Norm":

                L_local_specification = [AL_specification, C]

            elif AL_specification == "Ensemble-Huber":

                L_local_specification = [AL_specification, v0, C]

            adaptive_learning = Huber_AL.adaptive_learning(k,
                                                           data_and_time=data,
                                                           csv_directory=f'../../../../../../DataCentre/AdaptiveLearning/Real_data/Forecasts/VAR_2x/',
                                                           functional_sets=functional_sets,
                                                           ar_orders=ar_orders,
                                                           window_sizes=window_sizes,
                                                           L_local=L_local_specification,
                                                           L_global=L_global_specification,
                                                           )

            # Naming change for saving.
            if AL_specification == "Huber-Norm":
                L_local_specification = ["DMS-MV-Huber", C]
            elif AL_specification == "Ensemble-Huber":
                L_local_specification = ["Ensemble-MV-Huber", v0, C]

            helper.save_models_and_forecasts(
                adaptive_learning, L_global_specification, L_local_specification, k, functional_sets, 1, False, method)
