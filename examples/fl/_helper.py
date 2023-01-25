from mldesigner import command_component, Output
from mldesigner._constants import AssetTypes


@command_component
def increase_iteration_number_component(
    iteration_num: int,
) -> Output(type="integer", is_control=True):
    return iteration_num + 1


@command_component
def true_output_component() -> Output(type="boolean", is_control=True):
    return True


@command_component
def aggregate_models(aggregated_output: Output(type=AssetTypes.MLTABLE), **kwargs):
    print("Aggregated Output {}".format(aggregated_output))
    for k, v in kwargs.items():
        print("Input name is {}".format(k))
        print("Input value is {}".format(v))

    print(f"Writing output to {aggregated_output}/aggregate.txt")
    import os
    with open(os.path.join(aggregated_output, "aggregate.txt"), "w") as f:
        f.write("Hello World!")

    mltable_yaml = """paths: 
    - file: ./titanic.csv
    """
    with open(os.path.join(aggregated_output, "MLTable"), "w") as f:
        f.write(mltable_yaml)


