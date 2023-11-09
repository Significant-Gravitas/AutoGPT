try:
    import wandb

    WANDB_AVAILABLE = True
except:
    WANDB_AVAILABLE = False


if WANDB_AVAILABLE:
    import datetime
    import io
    import json
    import re
    from pathlib import Path

    from openai import File, FineTune
    from openai.datalib import numpy as np
    from openai.datalib import pandas as pd


class WandbLogger:
    """
    Log fine-tunes to [Weights & Biases](https://wandb.me/openai-docs)
    """

    if not WANDB_AVAILABLE:
        print("Logging requires wandb to be installed. Run `pip install wandb`.")
    else:
        _wandb_api = None
        _logged_in = False

    @classmethod
    def sync(
        cls,
        id=None,
        n_fine_tunes=None,
        project="GPT-3",
        entity=None,
        force=False,
        **kwargs_wandb_init,
    ):
        """
        Sync fine-tunes to Weights & Biases.
        :param id: The id of the fine-tune (optional)
        :param n_fine_tunes: Number of most recent fine-tunes to log when an id is not provided. By default, every fine-tune is synced.
        :param project: Name of the project where you're sending runs. By default, it is "GPT-3".
        :param entity: Username or team name where you're sending runs. By default, your default entity is used, which is usually your username.
        :param force: Forces logging and overwrite existing wandb run of the same fine-tune.
        """

        if not WANDB_AVAILABLE:
            return

        if id:
            fine_tune = FineTune.retrieve(id=id)
            fine_tune.pop("events", None)
            fine_tunes = [fine_tune]

        else:
            # get list of fine_tune to log
            fine_tunes = FineTune.list()
            if not fine_tunes or fine_tunes.get("data") is None:
                print("No fine-tune has been retrieved")
                return
            fine_tunes = fine_tunes["data"][
                -n_fine_tunes if n_fine_tunes is not None else None :
            ]

        # log starting from oldest fine_tune
        show_individual_warnings = (
            False if id is None and n_fine_tunes is None else True
        )
        fine_tune_logged = [
            cls._log_fine_tune(
                fine_tune,
                project,
                entity,
                force,
                show_individual_warnings,
                **kwargs_wandb_init,
            )
            for fine_tune in fine_tunes
        ]

        if not show_individual_warnings and not any(fine_tune_logged):
            print("No new successful fine-tunes were found")

        return "🎉 wandb sync completed successfully"

    @classmethod
    def _log_fine_tune(
        cls,
        fine_tune,
        project,
        entity,
        force,
        show_individual_warnings,
        **kwargs_wandb_init,
    ):
        fine_tune_id = fine_tune.get("id")
        status = fine_tune.get("status")

        # check run completed successfully
        if status != "succeeded":
            if show_individual_warnings:
                print(
                    f'Fine-tune {fine_tune_id} has the status "{status}" and will not be logged'
                )
            return

        # check results are present
        try:
            results_id = fine_tune["result_files"][0]["id"]
            results = File.download(id=results_id).decode("utf-8")
        except:
            if show_individual_warnings:
                print(f"Fine-tune {fine_tune_id} has no results and will not be logged")
            return

        # check run has not been logged already
        run_path = f"{project}/{fine_tune_id}"
        if entity is not None:
            run_path = f"{entity}/{run_path}"
        wandb_run = cls._get_wandb_run(run_path)
        if wandb_run:
            wandb_status = wandb_run.summary.get("status")
            if show_individual_warnings:
                if wandb_status == "succeeded":
                    print(
                        f"Fine-tune {fine_tune_id} has already been logged successfully at {wandb_run.url}"
                    )
                    if not force:
                        print(
                            'Use "--force" in the CLI or "force=True" in python if you want to overwrite previous run'
                        )
                else:
                    print(
                        f"A run for fine-tune {fine_tune_id} was previously created but didn't end successfully"
                    )
                if wandb_status != "succeeded" or force:
                    print(
                        f"A new wandb run will be created for fine-tune {fine_tune_id} and previous run will be overwritten"
                    )
            if wandb_status == "succeeded" and not force:
                return

        # start a wandb run
        wandb.init(
            job_type="fine-tune",
            config=cls._get_config(fine_tune),
            project=project,
            entity=entity,
            name=fine_tune_id,
            id=fine_tune_id,
            **kwargs_wandb_init,
        )

        # log results
        df_results = pd.read_csv(io.StringIO(results))
        for _, row in df_results.iterrows():
            metrics = {k: v for k, v in row.items() if not np.isnan(v)}
            step = metrics.pop("step")
            if step is not None:
                step = int(step)
            wandb.log(metrics, step=step)
        fine_tuned_model = fine_tune.get("fine_tuned_model")
        if fine_tuned_model is not None:
            wandb.summary["fine_tuned_model"] = fine_tuned_model

        # training/validation files and fine-tune details
        cls._log_artifacts(fine_tune, project, entity)

        # mark run as complete
        wandb.summary["status"] = "succeeded"

        wandb.finish()
        return True

    @classmethod
    def _ensure_logged_in(cls):
        if not cls._logged_in:
            if wandb.login():
                cls._logged_in = True
            else:
                raise Exception("You need to log in to wandb")

    @classmethod
    def _get_wandb_run(cls, run_path):
        cls._ensure_logged_in()
        try:
            if cls._wandb_api is None:
                cls._wandb_api = wandb.Api()
            return cls._wandb_api.run(run_path)
        except Exception:
            return None

    @classmethod
    def _get_wandb_artifact(cls, artifact_path):
        cls._ensure_logged_in()
        try:
            if cls._wandb_api is None:
                cls._wandb_api = wandb.Api()
            return cls._wandb_api.artifact(artifact_path)
        except Exception:
            return None

    @classmethod
    def _get_config(cls, fine_tune):
        config = dict(fine_tune)
        for key in ("training_files", "validation_files", "result_files"):
            if config.get(key) and len(config[key]):
                config[key] = config[key][0]
        if config.get("created_at"):
            config["created_at"] = datetime.datetime.fromtimestamp(config["created_at"])
        return config

    @classmethod
    def _log_artifacts(cls, fine_tune, project, entity):
        # training/validation files
        training_file = (
            fine_tune["training_files"][0]
            if fine_tune.get("training_files") and len(fine_tune["training_files"])
            else None
        )
        validation_file = (
            fine_tune["validation_files"][0]
            if fine_tune.get("validation_files") and len(fine_tune["validation_files"])
            else None
        )
        for file, prefix, artifact_type in (
            (training_file, "train", "training_files"),
            (validation_file, "valid", "validation_files"),
        ):
            if file is not None:
                cls._log_artifact_inputs(file, prefix, artifact_type, project, entity)

        # fine-tune details
        fine_tune_id = fine_tune.get("id")
        artifact = wandb.Artifact(
            "fine_tune_details",
            type="fine_tune_details",
            metadata=fine_tune,
        )
        with artifact.new_file(
            "fine_tune_details.json", mode="w", encoding="utf-8"
        ) as f:
            json.dump(fine_tune, f, indent=2)
        wandb.run.log_artifact(
            artifact,
            aliases=["latest", fine_tune_id],
        )

    @classmethod
    def _log_artifact_inputs(cls, file, prefix, artifact_type, project, entity):
        file_id = file["id"]
        filename = Path(file["filename"]).name
        stem = Path(file["filename"]).stem

        # get input artifact
        artifact_name = f"{prefix}-{filename}"
        # sanitize name to valid wandb artifact name
        artifact_name = re.sub(r"[^a-zA-Z0-9_\-.]", "_", artifact_name)
        artifact_alias = file_id
        artifact_path = f"{project}/{artifact_name}:{artifact_alias}"
        if entity is not None:
            artifact_path = f"{entity}/{artifact_path}"
        artifact = cls._get_wandb_artifact(artifact_path)

        # create artifact if file not already logged previously
        if artifact is None:
            # get file content
            try:
                file_content = File.download(id=file_id).decode("utf-8")
            except:
                print(
                    f"File {file_id} could not be retrieved. Make sure you are allowed to download training/validation files"
                )
                return
            artifact = wandb.Artifact(artifact_name, type=artifact_type, metadata=file)
            with artifact.new_file(filename, mode="w", encoding="utf-8") as f:
                f.write(file_content)

            # create a Table
            try:
                table, n_items = cls._make_table(file_content)
                artifact.add(table, stem)
                wandb.config.update({f"n_{prefix}": n_items})
                artifact.metadata["items"] = n_items
            except:
                print(f"File {file_id} could not be read as a valid JSON file")
        else:
            # log number of items
            wandb.config.update({f"n_{prefix}": artifact.metadata.get("items")})

        wandb.run.use_artifact(artifact, aliases=["latest", artifact_alias])

    @classmethod
    def _make_table(cls, file_content):
        df = pd.read_json(io.StringIO(file_content), orient="records", lines=True)
        return wandb.Table(dataframe=df), len(df)
