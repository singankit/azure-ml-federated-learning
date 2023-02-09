import pathlib
import shutil
from pathlib import Path

from mldesigner import command_component, Output, Input


# @command_component
# def increase_iteration_number_component(
#     iteration_num: int,
# ) -> Output(type="integer", is_control=True):
#     return iteration_num + 1
#
#
# @command_component
# def true_output_component() -> Output(type="boolean", is_control=True):
#     return True

def save_mltable_yaml(path, mltable_paths):
    import os
    import yaml
    path = os.path.abspath(path)

    if os.path.isfile(path):
        raise ValueError(f'The given path {path} points to a file.')

    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    save_path = os.path.join(path, 'MLTable')

    # mltable_yaml_dict = yaml.dump({"paths": mltable_paths}, default_)
    # mltable_yaml_dict[_PATHS_KEY] = self.paths
    with open(save_path, 'w') as f:
        yaml.dump({"paths": mltable_paths}, f)


@command_component(environment="azureml:test_fl_command_component@latest")
def aggregate_models(aggregated_output: Output(type="mltable"), **kwargs):
    print("Aggregated Output {}".format(aggregated_output))
    path_list = []
    for k, v in kwargs.items():
        print("Input name is {}".format(k))
        print("Input value is {}".format(v))
        path_list.append({"folder": v})

    save_mltable_yaml(aggregated_output, path_list)

    # print(f"Writing output to {aggregated_output}/aggregate.txt")
    # import os
    # with open(os.path.join(aggregated_output, "aggregate.txt"), "w") as f:
    #     f.write("Hello World!")
    #
    # mltable_yaml = """paths:
    # - file: ./titanic.csv
    # """
    # with open(os.path.join(aggregated_output, "MLTable"), "w") as f:
    #     f.write(mltable_yaml)


def get_model_locations(aggregated_model):
    import os
    for root, dirs, _ in os.walk(aggregated_model):
        # target: /mnt/azureml/.../${{default_datastore}}/azureml/${{name}}/${{output_name}}
        # we are looking for the level ${{name}}, there should be more than one directory.
        if len(dirs) > 1:
            return Path(root)


@command_component
def aggregator(aggregated_model: Input(type="mltable"), final_model: Output, **kwargs):
    print(aggregated_model)
    import os
    print(os.listdir(aggregated_model))
    models_location = get_model_locations(aggregated_model)
    print("Model location is {}".format(models_location))

    for silo_num in os.listdir(models_location):
        print("Silo number is {}".format(silo_num))
        for job_name in os.listdir(pathlib.Path.joinpath(models_location, silo_num)):
            print("Job name in {}".format(job_name))
            print("Full path to directory containing model is {}".format(str(
                pathlib.Path.joinpath(models_location, silo_num, job_name)
            )))
            for iteration_num in os.listdir(pathlib.Path.joinpath(models_location, silo_num, job_name)):
                print("Copying data from {} to {}". format(
                    str(pathlib.Path.joinpath(models_location, silo_num, job_name, iteration_num)),
                    str(pathlib.Path.joinpath(Path(final_model), "{}_{}".format(silo_num, iteration_num)))
                ))
                shutil.copytree(
                    pathlib.Path.joinpath(models_location, silo_num, job_name, iteration_num),
                    pathlib.Path.joinpath(Path(final_model), "{}_{}".format(silo_num, iteration_num))
                )

    # print(f"Writing output to {final_model}/final_model.txt")
    # import os
    # with open(os.path.join(final_model, "final_model.txt"), "w") as f:
    #     f.write("Final Model Hello World!")
