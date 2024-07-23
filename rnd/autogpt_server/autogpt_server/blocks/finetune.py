from typing import Any, List
import json
import openai
from autogpt_server.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from autogpt_server.data.model import BlockSecret, SecretField


class FinetuneBlock(Block):
    class Input(BlockSchema):
        api_key: BlockSecret = SecretField(key="openai_api_key", description="OpenAI API key")
        model: str = "gpt-3.5-turbo"
        training_data: str
        validation_split: float = 0.2
        n_epochs: int = 3
        batch_size: int = 1

    class Output(BlockSchema):
        job_id: str
        status: str
        error: str

    def __init__(self):
        super().__init__(
            id="b9a8c7d6-e5f4-3g2h-1i0j-k9l8m7n6o5p4",
            description="Create and start an OpenAI fine-tuning job with JSONL formatted data",
            categories={BlockCategory.LLM, BlockCategory.TRAINING},
            input_schema=FinetuneBlock.Input,
            output_schema=FinetuneBlock.Output,
            test_input={
                "api_key": "sk-test123",
                "model": "gpt-3.5-turbo",
                "training_data": '{"messages": [{"role": "system", "content": "Marv is a factual chatbot that is also sarcastic."}, {"role": "user", "content": "What\'s the capital of France?"}, {"role": "assistant", "content": "Paris, as if everyone doesn\'t know that already."}]}\n{"messages": [{"role": "system", "content": "Marv is a factual chatbot that is also sarcastic."}, {"role": "user", "content": "Who wrote \'Romeo and Juliet\'?"}, {"role": "assistant", "content": "Oh, just some guy named William Shakespeare. Ever heard of him?"}]}'
            },
            test_output=[("job_id", "ft-abc123"), ("status", "created")],
            test_mock={"create_fine_tuning_job": lambda *args, **kwargs: {"id": "ft-abc123", "status": "created"}}
        )

    @staticmethod
    def split_data(data: str, validation_split: float) -> tuple[str, str]:
        lines = data.strip().split('\n')
        split_index = int(len(lines) * (1 - validation_split))
        return '\n'.join(lines[:split_index]), '\n'.join(lines[split_index:])

    @staticmethod
    def create_fine_tuning_job(api_key: str, model: str, training_data: str, validation_data: str, n_epochs: int,
                               batch_size: int) -> dict:
        openai.api_key = api_key

        training_file = openai.File.create(
            file=training_data,
            purpose='fine-tune'
        )

        job_params = {
            "training_file": training_file.id,
            "model": model,
            "hyperparameters": {
                "n_epochs": n_epochs,
                "batch_size": batch_size
            }
        }

        if validation_data:
            validation_file = openai.File.create(
                file=validation_data,
                purpose='fine-tune'
            )
            job_params["validation_file"] = validation_file.id

        job = openai.FineTuningJob.create(**job_params)
        return job

    def run(self, input_data: Input) -> BlockOutput:
        try:
            # Split data into training and validation sets
            training_data, validation_data = self.split_data(input_data.training_data, input_data.validation_split)

            job = self.create_fine_tuning_job(
                api_key=input_data.api_key.get_secret_value(),
                model=input_data.model,
                training_data=training_data,
                validation_data=validation_data,
                n_epochs=input_data.n_epochs,
                batch_size=input_data.batch_size
            )

            yield "job_id", job.id
            yield "status", job.status
        except Exception as e:
            yield "error", str(e)
