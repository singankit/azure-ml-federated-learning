import os

os.environ["AZURE_ML_CLI_PRIVATE_FEATURES_ENABLED"] = "True"

from azure.ai.ml import Input, load_component
from azure.ai.ml.constants import AssetTypes

# path to the components
from azure.ai.ml.dsl import pipeline

COMPONENTS_FOLDER = os.path.join(
    os.path.dirname(__file__), "..", "..", "components", "HELLOWORLD"
)

# path to the shared components
SHARED_COMPONENTS_FOLDER = os.path.join(
    os.path.dirname(__file__), "..", "components", "utils"
)

####################################
### LOAD THE PIPELINE COMPONENTS ###
####################################

# Loading the component from their yaml specifications
preprocessing_component = load_component(
    source=os.path.join(COMPONENTS_FOLDER, "preprocessing", "spec.yaml")
)

training_component = load_component(
    source=os.path.join(COMPONENTS_FOLDER, "traininsilo", "spec.yaml")
)

aggregate_component = load_component(
    source=os.path.join(COMPONENTS_FOLDER, "aggregatemodelweights", "spec.yaml")
)

# evaluation_component = load_component(
#     source=os.path.join(COMPONENTS_FOLDER, "evaluate", "spec.yaml")
# )


def get_silo_configs():
    silo_configs = [
        {
            "compute": "cpu-cluster-westeurope",
            "datastore": "workspaceblobstorewesteurope",
            "inputs": {
                # feeds into the user defined inputs
                "raw_train_data": Input(
                    type=AssetTypes.URI_FILE,
                    mode="download",
                    path="https://azureopendatastorage.blob.core.windows.net/mnist/processed/train.csv",
                ),
                "raw_test_data": Input(
                    type=AssetTypes.URI_FILE,
                    mode="download",
                    path="https://azureopendatastorage.blob.core.windows.net/mnist/processed/t10k.csv",
                )},
        },
        {
            "compute": "cpu-cluster-australiaeast",
            "datastore": "workspaceblobstoreaustraliaeast",
            "inputs": {
                # feeds into the user defined inputs
                "raw_train_data": Input(
                    type=AssetTypes.URI_FILE,
                    mode="download",
                    path="https://azureopendatastorage.blob.core.windows.net/mnist/processed/train.csv",
                ),
                "raw_test_data": Input(
                    type=AssetTypes.URI_FILE,
                    mode="download",
                    path="https://azureopendatastorage.blob.core.windows.net/mnist/processed/t10k.csv",
                )},
        }
    ]

    return silo_configs


def get_gather_config():
    return {
        "compute": "cpu-cluster-1",
        "datastore": "flgathermodels",
        }


@pipeline
def gather_pipeline(
    input_silo_1: Input,
    input_silo_2: Input,
):
    gather_test = aggregate_component(
        input_silo_1=input_silo_1,
        input_silo_2=input_silo_2)
    return {
            "aggregated_output": gather_test.outputs.aggregated_output
        }


@pipeline(
    name="Silo Federated Learning Subgraph",
    description="It includes preprocessing, training and local evaluation",
)
def silo_scatter_subgraph(
    # user defined inputs
    raw_train_data: Input,
    raw_test_data: Input,
    # raw_eval_data: Input,
    # user defined accumulator
    checkpoint: Input(optional=True),
    # RESERVED arguments
    # we propose that the SDK provides those arguments to the subgraph
    # to help with building the graph
    scatter_compute: str,
    # scatter_datastore: str,
    # gather_datastore: str,
    iteration_num: int,
    # user defined inputs
    lr: float = 0.01,
    epochs: int = 3,
    batch_size: int = 64,
):
    """Create silo/training subgraph.
    Args:
        raw_train_data (Input): raw train data
        raw_test_data (Input): raw test data
        raw_eval_data (Input): raw test data
        checkpoint (Input): if not None, the checkpoint obtained from previous iteration (see orchestrator_aggregation())
        lr (float, optional): Learning rate. Defaults to 0.01.
        epochs (int, optional): Number of epochs. Defaults to 3.
        batch_size (int, optional): Batch size. Defaults to 64.
        scatter_compute (str): Silo compute name
        scatter_datastore (str): Silo datastore name
        gather_datastore (str): Orchestrator datastore name
        iteration_num (int): Iteration number
    Returns:
        Dict[str, Outputs]: a map of the outputs
    """
    # we're using our own preprocessing component
    silo_pre_processing_step = preprocessing_component(
        # this consumes whatever user defined inputs
        raw_training_data=raw_train_data,
        raw_testing_data=raw_test_data,
        # raw_evaluation_data=raw_eval_data,
        # here we're using the name of the silo compute as a metrics prefix
        metrics_prefix=scatter_compute,
    )

    # we're using our own training component
    silo_training_step = training_component(
        # with the train_data from the pre_processing step
        train_data=silo_pre_processing_step.outputs.processed_train_data,
        # with the test_data from the pre_processing step
        test_data=silo_pre_processing_step.outputs.processed_test_data,
        # and the checkpoint from previous iteration (or None if iteration == 1)
        checkpoint=checkpoint,
        # Learning rate for local training
        lr=lr,
        # Number of epochs
        epochs=epochs,
        # Dataloader batch size
        batch_size=batch_size,
        # Silo name/identifier
        metrics_prefix=scatter_compute,
        # Iteration number
        iteration_num=iteration_num,
    )

    # we even have an evaluation step
    # silo_evaluation_step = evaluation_component(
    #     model=silo_training_step.outputs.model,
    #     eval_data=silo_pre_processing_step.outputs.processed_eval_data,
    # )

    # IMPORTANT: we will assume that any output provided here can be exfiltrated into the orchestrator
    # the SDK will assign this output to the ORCHESTRATOR datastore
    return {
        # NOTE: the key you use is custom
        # a map function scatter_to_gather_map needs to be provided
        # to map the name here to the expected input from gather
        "model": silo_training_step.outputs.model,
        # "metrics": silo_evaluation_step.outputs.metrics,
    }


def main():
    silo_configs = get_silo_configs()
    gather_config = get_gather_config()
    from fl_pipeline import scatter_gather_iteration

    pipeline_fl = scatter_gather_iteration(
        scatter=silo_scatter_subgraph,
        gather=aggregate_component,
        # gather=gather_pipeline,
        scatter_strategy=silo_configs,
        gather_strategy=gather_config,
        scatter_constant_inputs={"lr": 0.01, "batch_size": 32, "epochs": 3},
        scatter_to_gather_map=lambda output_name, silo_index: f"input_silo_{silo_index}",
        iterations=2,
    )

    from azure.ai.ml import MLClient
    from azure.identity import DefaultAzureCredential
    ml_client = MLClient(
        subscription_id="b17253fa-f327-42d6-9686-f3e553e24763",
        resource_group_name="anksing-rg",
        workspace_name="anksing-wcus",
        credential=DefaultAzureCredential()
    )

    # pipeline_fl.settings.default_compute = "cpu-cluster"
    ml_client.jobs.create_or_update(pipeline_fl)


if __name__ == "__main__":
    main()
