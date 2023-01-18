from mldesigner import command_component, Output


@command_component
def increase_iteration_number_component(
    iteration_num: int,
) -> Output(type="integer", is_control=True):
    return iteration_num + 1


@command_component
def true_output_component() -> Output(type="boolean", is_control=True):
    return True
