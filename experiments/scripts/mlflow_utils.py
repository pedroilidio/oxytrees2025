import os
import re
from pathlib import Path

import mlflow.tracking


def uri_to_path(uri: str) -> Path:
    """Convert a URI to a Path object."""
    if not re.match(r"^[^:^/]+?:/", uri):
        return Path(uri).resolve()
    if uri.startswith("file://"):
        # TODO: Python 3.13 has Path.from_uri()
        return Path(uri.removeprefix("file://")).resolve()
    raise ValueError(f"Unsupported URI format: {uri}")


def get_experiment_id_from_name(
    *, client: mlflow.tracking.MlflowClient, experiment_name: str, description: str
) -> str:
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"Creating experiment: {experiment_name}")
        return client.create_experiment(
            name=experiment_name,
            # artifact_location=None,  # TODO: Specify artifact location
            tags={"mlflow.note.content": description},
        )
    print(f"Found existing experiment: {vars(experiment)}")

    if experiment.lifecycle_stage == "active":
        try:
            path_artifacts = uri_to_path(experiment.artifact_location)
        except ValueError:
            print("Remote artifact location. Using existing experiment.")
            return experiment.experiment_id

        if os.access(path_artifacts, os.W_OK):
            print("Using existing experiment.")
            return experiment.experiment_id

        print(
            f"Artifact_location is not writable: {experiment.artifact_location}"
            f" (Path: {path_artifacts})"
        )

    # HACK
    sep = "__v"
    if sep in experiment_name:
        # If the experiment name already contains a version number, increment it.
        experiment_name, version = experiment_name.rsplit(sep, maxsplit=1)
        try:
            version = str(int(version) + 1)
        except ValueError:
            version += sep + "2"
    else:
        version = "2"

    new_name = experiment_name + sep + version

    print(f"Trying experiment name: {new_name}")
    return get_experiment_id_from_name(
        client=client,
        experiment_name=new_name,
        description=description,
    )


def log_sklearn_model(
    client: mlflow.tracking.MlflowClient,
    run_id: str,
    estimator,
    code_path: str,
):
    # TODO
    # mlflow.models.Model.log(
    #     run_id=run_id,
    #     sk_model=estimator,
    #     artifact_path="model",
    #     flavor=mlflow.sklearn,
    #     signature=BIPARTITE_SIGNATURE,
    #     code_paths=[code_path],
    #     # TODO: Tensors are currently not supported by MLflow
    #     # input_example=[X[0][:2], X[1][:3]],
    # )
    str_params = {k: str(v) for k, v in estimator.get_params().items()}  # FIXME
    client.log_dict(run_id, str_params, "estimator_params.yml")
    client.log_param(run_id, "estimator_class", estimator.__class__.__name__)
    client.log_param(run_id, "estimator_module", code_path)
