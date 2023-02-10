from azure.ai.ml import Output, Input
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.constants import AssetTypes

from azure.ai.ml.dsl._do_while import do_while
from _helper import increase_iteration_number_component, true_output_component


def custom_fl_data_path(
    datastore_name, output_name, unique_id="${{name}}", iteration_num=None
):
    """Produces a path to store the data during FL training.
    Args:
        datastore_name (str): name of the Azure ML datastore
        output_name (str): a name unique to this output
        unique_id (str): a unique id for the run (default: inject run id with ${{name}})
        iteration_num (str): an iteration number if relevant
    Returns:
        data_path (str): direct url to the data path to store the data
    """
    data_path = f"azureml://datastores/{datastore_name}/paths/federated_learning/{output_name}/{unique_id}/"
    if iteration_num is not None:
        data_path += f"iteration_{iteration_num}/"

    return data_path


def scatter_gather_iteration(
    scatter,
    gather,
    scatter_strategy,
    gather_strategy,
    scatter_to_gather_map=None,
    gather_to_scatter_map=None,
    accumulators=None,
    iterations=None,
    scatter_constant_inputs=None,
):
    @pipeline(name="Scatter Gather Iteration")
    # def scatter_gather_iter(iteration_num: int, checkpoint: Input(type="uri_folder", optional=True)):
    def scatter_gather_iter(**kwargs):
        iteration_num = kwargs.pop("iteration_num")
        gather_inputs = {}
        for silo_index in range(1, len(scatter_strategy) + 1):
            silo_config = scatter_strategy[silo_index - 1]

            scatter_input = {
                **silo_config["inputs"], **scatter_constant_inputs,
                **{"scatter_compute": silo_config["compute"]}
            }

            # scatter_gather = scatter(**scatter_input, iteration_num=iteration_num, checkpoint=checkpoint)
            # scatter_gather = scatter(**scatter_input, checkpoint=checkpoint)
            scatter_gather = scatter(**scatter_input, iteration_num=iteration_num, **kwargs)

            # TODO: Assumption that scatter subgraph produces output with name "model"
            scatter_gather.outputs.model = Output(
                type=AssetTypes.URI_FOLDER,
                mode="mount",
                path=custom_fl_data_path(
                    gather_strategy["datastore"],
                    f"model/silo{silo_index}",
                    iteration_num=iteration_num.result()  # change this once iteration loop is added
                ),
            )

            # TODO: Hack for POC should not be used for prod code
            # Check with pipeline team on support for overriding compute and datastore for subgraphs
            # Until then compute need to be explicitly passed to child components as a param
            # This adds requirement that components used in FL should accept compute as param and
            # handle it appropriately
            for job_name, job in scatter_gather.component.jobs.items():
                job.compute = silo_config["compute"]

            # TODO: scatter_to_gather_map is user provided and no good way to know what inputs needed
            # What is the expectation for it ?
            gather_inputs[scatter_to_gather_map("model", silo_index)] = scatter_gather.outputs.model

        gather_instance = gather(**gather_inputs)
        # gather_instance = gather(**gather_inputs)
        gather_instance.outputs.aggregated_output = Output(
            type=AssetTypes.URI_FOLDER,
            mode="mount",
            path=custom_fl_data_path(
                gather_strategy["datastore"],
                f"model_aggregated",
                iteration_num=iteration_num.result()
            ),
        )

        # TODO: Hack for POC should not be used for prod code
        # Check with pipeline team on support for overriding compute and datastore for subgraphs
        # Until then compute need to be explicity passed to child components as a param
        # This adds requirement that components used in FL should accept compute as param and
        # handle it appropriately
        gather_instance.compute = gather_strategy["compute"]

        # for job_name, job in gather_instance.component.jobs.items():
        #     job.compute = gather_strategy["compute"]

        increase_iteration_number_component_instance = increase_iteration_number_component(
            iteration_num=iteration_num
        )
        increase_iteration_number_component_instance.compute = silo_config["compute"]

        true_output_component_instance = true_output_component()
        true_output_component_instance.compute = silo_config["compute"]

        return {
            # Assuming gather should return aggregated_model as output(Need to change in this component)
            "model": gather_instance.outputs.aggregated_output,
            "next_iteration_num": increase_iteration_number_component_instance.outputs.output,
            "early_terminate": true_output_component_instance.outputs.output,
        }

    # return scatter_gather_iter(checkpoint=None)

        # for iteration_number in range(0, 1): # Have to remove this so that
        #     gather_inputs = {}
        #     for silo_index in range(1, len(scatter_strategy) + 1):
        #         silo_config = scatter_strategy[silo_index - 1]
        #
        #         scatter_input = {
        #             **silo_config["inputs"], **scatter_constant_inputs,
        #             **{"scatter_compute": silo_config["compute"]}
        #         }
        #
        #         scatter_gather = scatter(**scatter_input, iteration_num=iteration_num)
        #
        #         # TODO: Assumption that scatter subgraph produces output with name "model"
        #         scatter_gather.outputs.model = Output(
        #             type=AssetTypes.URI_FOLDER,
        #             mode="mount",
        #             path=custom_fl_data_path(
        #                 gather_strategy["datastore"],
        #                 f"model/silo{silo_index}",
        #                 iteration_num=iteration_num  # change this once iteration loop is added
        #             ),
        #         )
        #
        #         # TODO: Hack for POC should not be used for prod code
        #         # Check with pipeline team on support for overriding compute and datastore for subgraphs
        #         # Until then compute need to be explicitly passed to child components as a param
        #         # This adds requirement that components used in FL should accept compute as param and
        #         # handle it appropriately
        #         for job_name, job in scatter_gather.component.jobs.items():
        #             job.compute = silo_config["compute"]
        #
        #         # TODO: scatter_to_gather_map is user provided and no good way to know what inputs needed
        #         # What is the expectation for it ?
        #         gather_inputs[scatter_to_gather_map("model", silo_index)] = scatter_gather.outputs.model
        #
        #     print(gather_inputs)
        #     gather_instance = gather(**gather_inputs)
        #     gather_instance.outputs.aggregated_output = Output(
        #         type=AssetTypes.URI_FOLDER,
        #         mode="mount",
        #         path=custom_fl_data_path(
        #             gather_strategy["datastore"],
        #             f"model/aggregated",
        #         ),
        #     )
        #     # TODO: Hack for POC should not be used for prod code
        #     # Check with pipeline team on support for overriding compute and datastore for subgraphs
        #     # Until then compute need to be explicitly passed to child components as a param
        #     # This adds requirement that components used in FL should accept compute as param and
        #     # handle it appropriately
        #     gather_instance.compute = gather_strategy["compute"]

    @pipeline(name="FL Pipeline with Do while")
    def fl_pipeline():
        iteration_body = scatter_gather_iter(iteration_num=0, checkpoint=None)

        do_while_mapping = {
            # user defined output(s) to input(s)
            "model": iteration_body.inputs.checkpoint,
            "next_iteration_num": iteration_body.inputs.iteration_num,
        }

        do_while_node = do_while(  # noqa: F841
            body=iteration_body,
            mapping=do_while_mapping,
            max_iteration_count=2,
            condition=iteration_body.outputs.early_terminate, # do_while supports early terminate
        )

        return {
            "final_model": iteration_body.outputs.model,
        }

    fl_pipeline_instance = fl_pipeline()
    fl_pipeline_instance.outputs.final_model = Output(
            type=AssetTypes.URI_FOLDER,
            mode="mount",
            path=custom_fl_data_path(
                gather_strategy["datastore"],
                f"final_model",
            ),
        )

    return fl_pipeline_instance

