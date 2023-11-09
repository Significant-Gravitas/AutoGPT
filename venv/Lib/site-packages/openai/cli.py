import datetime
import os
import signal
import sys
import warnings
from typing import Optional

import requests

import openai
from openai.upload_progress import BufferReader
from openai.validators import (
    apply_necessary_remediation,
    apply_validators,
    get_validators,
    read_any_format,
    write_out_file,
)


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def organization_info(obj):
    organization = getattr(obj, "organization", None)
    if organization is not None:
        return "[organization={}] ".format(organization)
    else:
        return ""


def display(obj):
    sys.stderr.write(organization_info(obj))
    sys.stderr.flush()
    print(obj)


def display_error(e):
    extra = (
        " (HTTP status code: {})".format(e.http_status)
        if e.http_status is not None
        else ""
    )
    sys.stderr.write(
        "{}{}Error:{} {}{}\n".format(
            organization_info(e), bcolors.FAIL, bcolors.ENDC, e, extra
        )
    )


class Engine:
    @classmethod
    def get(cls, args):
        engine = openai.Engine.retrieve(id=args.id)
        display(engine)

    @classmethod
    def update(cls, args):
        engine = openai.Engine.modify(args.id, replicas=args.replicas)
        display(engine)

    @classmethod
    def generate(cls, args):
        warnings.warn(
            "Engine.generate is deprecated, use Completion.create", DeprecationWarning
        )
        if args.completions and args.completions > 1 and args.stream:
            raise ValueError("Can't stream multiple completions with openai CLI")

        kwargs = {}
        if args.model is not None:
            kwargs["model"] = args.model
        resp = openai.Engine(id=args.id).generate(
            completions=args.completions,
            context=args.context,
            length=args.length,
            stream=args.stream,
            temperature=args.temperature,
            top_p=args.top_p,
            logprobs=args.logprobs,
            stop=args.stop,
            **kwargs,
        )
        if not args.stream:
            resp = [resp]

        for part in resp:
            completions = len(part["data"])
            for c_idx, c in enumerate(part["data"]):
                if completions > 1:
                    sys.stdout.write("===== Completion {} =====\n".format(c_idx))
                sys.stdout.write("".join(c["text"]))
                if completions > 1:
                    sys.stdout.write("\n")
                sys.stdout.flush()

    @classmethod
    def list(cls, args):
        engines = openai.Engine.list()
        display(engines)


class ChatCompletion:
    @classmethod
    def create(cls, args):
        if args.n is not None and args.n > 1 and args.stream:
            raise ValueError(
                "Can't stream chat completions with n>1 with the current CLI"
            )

        messages = [
            {"role": role, "content": content} for role, content in args.message
        ]

        resp = openai.ChatCompletion.create(
            # Required
            model=args.model,
            messages=messages,
            # Optional
            n=args.n,
            max_tokens=100,
            temperature=args.temperature,
            top_p=args.top_p,
            stop=args.stop,
            stream=args.stream,
        )
        if not args.stream:
            resp = [resp]

        for part in resp:
            choices = part["choices"]
            for c_idx, c in enumerate(sorted(choices, key=lambda s: s["index"])):
                if len(choices) > 1:
                    sys.stdout.write("===== Chat Completion {} =====\n".format(c_idx))
                sys.stdout.write(c["message"]["content"])
                if len(choices) > 1:
                    sys.stdout.write("\n")
                sys.stdout.flush()


class Completion:
    @classmethod
    def create(cls, args):
        if args.n is not None and args.n > 1 and args.stream:
            raise ValueError("Can't stream completions with n>1 with the current CLI")

        if args.engine and args.model:
            warnings.warn(
                "In most cases, you should not be specifying both engine and model."
            )

        resp = openai.Completion.create(
            engine=args.engine,
            model=args.model,
            n=args.n,
            max_tokens=args.max_tokens,
            logprobs=args.logprobs,
            prompt=args.prompt,
            stream=args.stream,
            temperature=args.temperature,
            top_p=args.top_p,
            stop=args.stop,
            echo=True,
        )
        if not args.stream:
            resp = [resp]

        for part in resp:
            choices = part["choices"]
            for c_idx, c in enumerate(sorted(choices, key=lambda s: s["index"])):
                if len(choices) > 1:
                    sys.stdout.write("===== Completion {} =====\n".format(c_idx))
                sys.stdout.write(c["text"])
                if len(choices) > 1:
                    sys.stdout.write("\n")
                sys.stdout.flush()


class Deployment:
    @classmethod
    def get(cls, args):
        resp = openai.Deployment.retrieve(id=args.id)
        print(resp)

    @classmethod
    def delete(cls, args):
        model = openai.Deployment.delete(args.id)
        print(model)

    @classmethod
    def list(cls, args):
        models = openai.Deployment.list()
        print(models)

    @classmethod
    def create(cls, args):
        models = openai.Deployment.create(model=args.model, scale_settings={"scale_type": args.scale_type})
        print(models)


class Model:
    @classmethod
    def get(cls, args):
        resp = openai.Model.retrieve(id=args.id)
        print(resp)

    @classmethod
    def delete(cls, args):
        model = openai.Model.delete(args.id)
        print(model)

    @classmethod
    def list(cls, args):
        models = openai.Model.list()
        print(models)


class File:
    @classmethod
    def create(cls, args):
        with open(args.file, "rb") as file_reader:
            buffer_reader = BufferReader(file_reader.read(), desc="Upload progress")
        resp = openai.File.create(
            file=buffer_reader,
            purpose=args.purpose,
            user_provided_filename=args.file,
        )
        print(resp)

    @classmethod
    def get(cls, args):
        resp = openai.File.retrieve(id=args.id)
        print(resp)

    @classmethod
    def delete(cls, args):
        file = openai.File.delete(args.id)
        print(file)

    @classmethod
    def list(cls, args):
        file = openai.File.list()
        print(file)


class Image:
    @classmethod
    def create(cls, args):
        resp = openai.Image.create(
            prompt=args.prompt,
            size=args.size,
            n=args.num_images,
            response_format=args.response_format,
        )
        print(resp)

    @classmethod
    def create_variation(cls, args):
        with open(args.image, "rb") as file_reader:
            buffer_reader = BufferReader(file_reader.read(), desc="Upload progress")
        resp = openai.Image.create_variation(
            image=buffer_reader,
            size=args.size,
            n=args.num_images,
            response_format=args.response_format,
        )
        print(resp)

    @classmethod
    def create_edit(cls, args):
        with open(args.image, "rb") as file_reader:
            image_reader = BufferReader(file_reader.read(), desc="Upload progress")
        mask_reader = None
        if args.mask is not None:
            with open(args.mask, "rb") as file_reader:
                mask_reader = BufferReader(file_reader.read(), desc="Upload progress")
        resp = openai.Image.create_edit(
            image=image_reader,
            mask=mask_reader,
            prompt=args.prompt,
            size=args.size,
            n=args.num_images,
            response_format=args.response_format,
        )
        print(resp)


class Audio:
    @classmethod
    def transcribe(cls, args):
        with open(args.file, "rb") as r:
            file_reader = BufferReader(r.read(), desc="Upload progress")

        resp = openai.Audio.transcribe_raw(
            # Required
            model=args.model,
            file=file_reader,
            filename=args.file,
            # Optional
            response_format=args.response_format,
            language=args.language,
            temperature=args.temperature,
            prompt=args.prompt,
        )
        print(resp)

    @classmethod
    def translate(cls, args):
        with open(args.file, "rb") as r:
            file_reader = BufferReader(r.read(), desc="Upload progress")
        resp = openai.Audio.translate_raw(
            # Required
            model=args.model,
            file=file_reader,
            filename=args.file,
            # Optional
            response_format=args.response_format,
            language=args.language,
            temperature=args.temperature,
            prompt=args.prompt,
        )
        print(resp)


class FineTune:
    @classmethod
    def list(cls, args):
        resp = openai.FineTune.list()
        print(resp)

    @classmethod
    def _is_url(cls, file: str):
        return file.lower().startswith("http")

    @classmethod
    def _download_file_from_public_url(cls, url: str) -> Optional[bytes]:
        resp = requests.get(url)
        if resp.status_code == 200:
            return resp.content
        else:
            return None

    @classmethod
    def _maybe_upload_file(
        cls,
        file: Optional[str] = None,
        content: Optional[bytes] = None,
        user_provided_file: Optional[str] = None,
        check_if_file_exists: bool = True,
    ):
        # Exactly one of `file` or `content` must be provided
        if (file is None) == (content is None):
            raise ValueError("Exactly one of `file` or `content` must be provided")

        if content is None:
            assert file is not None
            with open(file, "rb") as f:
                content = f.read()

        if check_if_file_exists:
            bytes = len(content)
            matching_files = openai.File.find_matching_files(
                name=user_provided_file or f.name, bytes=bytes, purpose="fine-tune"
            )
            if len(matching_files) > 0:
                file_ids = [f["id"] for f in matching_files]
                sys.stdout.write(
                    "Found potentially duplicated files with name '{name}', purpose 'fine-tune' and size {size} bytes\n".format(
                        name=os.path.basename(matching_files[0]["filename"]),
                        size=matching_files[0]["bytes"]
                        if "bytes" in matching_files[0]
                        else matching_files[0]["size"],
                    )
                )
                sys.stdout.write("\n".join(file_ids))
                while True:
                    sys.stdout.write(
                        "\nEnter file ID to reuse an already uploaded file, or an empty string to upload this file anyway: "
                    )
                    inp = sys.stdin.readline().strip()
                    if inp in file_ids:
                        sys.stdout.write(
                            "Reusing already uploaded file: {id}\n".format(id=inp)
                        )
                        return inp
                    elif inp == "":
                        break
                    else:
                        sys.stdout.write(
                            "File id '{id}' is not among the IDs of the potentially duplicated files\n".format(
                                id=inp
                            )
                        )

        buffer_reader = BufferReader(content, desc="Upload progress")
        resp = openai.File.create(
            file=buffer_reader,
            purpose="fine-tune",
            user_provided_filename=user_provided_file or file,
        )
        sys.stdout.write(
            "Uploaded file from {file}: {id}\n".format(
                file=user_provided_file or file, id=resp["id"]
            )
        )
        return resp["id"]

    @classmethod
    def _get_or_upload(cls, file, check_if_file_exists=True):
        try:
            # 1. If it's a valid file, use it
            openai.File.retrieve(file)
            return file
        except openai.error.InvalidRequestError:
            pass
        if os.path.isfile(file):
            # 2. If it's a file on the filesystem, upload it
            return cls._maybe_upload_file(
                file=file, check_if_file_exists=check_if_file_exists
            )
        if cls._is_url(file):
            # 3. If it's a URL, download it temporarily
            content = cls._download_file_from_public_url(file)
            if content is not None:
                return cls._maybe_upload_file(
                    content=content,
                    check_if_file_exists=check_if_file_exists,
                    user_provided_file=file,
                )
        return file

    @classmethod
    def create(cls, args):
        create_args = {
            "training_file": cls._get_or_upload(
                args.training_file, args.check_if_files_exist
            ),
        }
        if args.validation_file:
            create_args["validation_file"] = cls._get_or_upload(
                args.validation_file, args.check_if_files_exist
            )

        for hparam in (
            "model",
            "suffix",
            "n_epochs",
            "batch_size",
            "learning_rate_multiplier",
            "prompt_loss_weight",
            "compute_classification_metrics",
            "classification_n_classes",
            "classification_positive_class",
            "classification_betas",
        ):
            attr = getattr(args, hparam)
            if attr is not None:
                create_args[hparam] = attr

        resp = openai.FineTune.create(**create_args)

        if args.no_follow:
            print(resp)
            return

        sys.stdout.write(
            "Created fine-tune: {job_id}\n"
            "Streaming events until fine-tuning is complete...\n\n"
            "(Ctrl-C will interrupt the stream, but not cancel the fine-tune)\n".format(
                job_id=resp["id"]
            )
        )
        cls._stream_events(resp["id"])

    @classmethod
    def get(cls, args):
        resp = openai.FineTune.retrieve(id=args.id)
        print(resp)

    @classmethod
    def results(cls, args):
        fine_tune = openai.FineTune.retrieve(id=args.id)
        if "result_files" not in fine_tune or len(fine_tune["result_files"]) == 0:
            raise openai.error.InvalidRequestError(
                f"No results file available for fine-tune {args.id}", "id"
            )
        result_file = openai.FineTune.retrieve(id=args.id)["result_files"][0]
        resp = openai.File.download(id=result_file["id"])
        print(resp.decode("utf-8"))

    @classmethod
    def events(cls, args):
        if args.stream:
            raise openai.error.OpenAIError(
                message=(
                    "The --stream parameter is deprecated, use fine_tunes.follow "
                    "instead:\n\n"
                    "  openai api fine_tunes.follow -i {id}\n".format(id=args.id)
                ),
            )

        resp = openai.FineTune.list_events(id=args.id)  # type: ignore
        print(resp)

    @classmethod
    def follow(cls, args):
        cls._stream_events(args.id)

    @classmethod
    def _stream_events(cls, job_id):
        def signal_handler(sig, frame):
            status = openai.FineTune.retrieve(job_id).status
            sys.stdout.write(
                "\nStream interrupted. Job is still {status}.\n"
                "To resume the stream, run:\n\n"
                "  openai api fine_tunes.follow -i {job_id}\n\n"
                "To cancel your job, run:\n\n"
                "  openai api fine_tunes.cancel -i {job_id}\n\n".format(
                    status=status, job_id=job_id
                )
            )
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)

        events = openai.FineTune.stream_events(job_id)
        # TODO(rachel): Add a nifty spinner here.
        try:
            for event in events:
                sys.stdout.write(
                    "[%s] %s"
                    % (
                        datetime.datetime.fromtimestamp(event["created_at"]),
                        event["message"],
                    )
                )
                sys.stdout.write("\n")
                sys.stdout.flush()
        except Exception:
            sys.stdout.write(
                "\nStream interrupted (client disconnected).\n"
                "To resume the stream, run:\n\n"
                "  openai api fine_tunes.follow -i {job_id}\n\n".format(job_id=job_id)
            )
            return

        resp = openai.FineTune.retrieve(id=job_id)
        status = resp["status"]
        if status == "succeeded":
            sys.stdout.write("\nJob complete! Status: succeeded 🎉")
            sys.stdout.write(
                "\nTry out your fine-tuned model:\n\n"
                "openai api completions.create -m {model} -p <YOUR_PROMPT>".format(
                    model=resp["fine_tuned_model"]
                )
            )
        elif status == "failed":
            sys.stdout.write(
                "\nJob failed. Please contact support@openai.com if you need assistance."
            )
        sys.stdout.write("\n")

    @classmethod
    def cancel(cls, args):
        resp = openai.FineTune.cancel(id=args.id)
        print(resp)

    @classmethod
    def delete(cls, args):
        resp = openai.FineTune.delete(sid=args.id)
        print(resp)

    @classmethod
    def prepare_data(cls, args):
        sys.stdout.write("Analyzing...\n")
        fname = args.file
        auto_accept = args.quiet
        df, remediation = read_any_format(fname)
        apply_necessary_remediation(None, remediation)

        validators = get_validators()

        apply_validators(
            df,
            fname,
            remediation,
            validators,
            auto_accept,
            write_out_file_func=write_out_file,
        )


class WandbLogger:
    @classmethod
    def sync(cls, args):
        import openai.wandb_logger

        resp = openai.wandb_logger.WandbLogger.sync(
            id=args.id,
            n_fine_tunes=args.n_fine_tunes,
            project=args.project,
            entity=args.entity,
            force=args.force,
        )
        print(resp)


def tools_register(parser):
    subparsers = parser.add_subparsers(
        title="Tools", help="Convenience client side tools"
    )

    def help(args):
        parser.print_help()

    parser.set_defaults(func=help)

    sub = subparsers.add_parser("fine_tunes.prepare_data")
    sub.add_argument(
        "-f",
        "--file",
        required=True,
        help="JSONL, JSON, CSV, TSV, TXT or XLSX file containing prompt-completion examples to be analyzed."
        "This should be the local file path.",
    )
    sub.add_argument(
        "-q",
        "--quiet",
        required=False,
        action="store_true",
        help="Auto accepts all suggestions, without asking for user input. To be used within scripts.",
    )
    sub.set_defaults(func=FineTune.prepare_data)


def api_register(parser):
    # Engine management
    subparsers = parser.add_subparsers(help="All API subcommands")

    def help(args):
        parser.print_help()

    parser.set_defaults(func=help)

    sub = subparsers.add_parser("engines.list")
    sub.set_defaults(func=Engine.list)

    sub = subparsers.add_parser("engines.get")
    sub.add_argument("-i", "--id", required=True)
    sub.set_defaults(func=Engine.get)

    sub = subparsers.add_parser("engines.update")
    sub.add_argument("-i", "--id", required=True)
    sub.add_argument("-r", "--replicas", type=int)
    sub.set_defaults(func=Engine.update)

    sub = subparsers.add_parser("engines.generate")
    sub.add_argument("-i", "--id", required=True)
    sub.add_argument(
        "--stream", help="Stream tokens as they're ready.", action="store_true"
    )
    sub.add_argument("-c", "--context", help="An optional context to generate from")
    sub.add_argument("-l", "--length", help="How many tokens to generate", type=int)
    sub.add_argument(
        "-t",
        "--temperature",
        help="""What sampling temperature to use. Higher values means the model will take more risks. Try 0.9 for more creative applications, and 0 (argmax sampling) for ones with a well-defined answer.

Mutually exclusive with `top_p`.""",
        type=float,
    )
    sub.add_argument(
        "-p",
        "--top_p",
        help="""An alternative to sampling with temperature, called nucleus sampling, where the considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10%% probability mass are considered.

            Mutually exclusive with `temperature`.""",
        type=float,
    )
    sub.add_argument(
        "-n",
        "--completions",
        help="How many parallel completions to run on this context",
        type=int,
    )
    sub.add_argument(
        "--logprobs",
        help="Include the log probabilites on the `logprobs` most likely tokens. So for example, if `logprobs` is 10, the API will return a list of the 10 most likely tokens. If `logprobs` is supplied, the API will always return the logprob of the generated token, so there may be up to `logprobs+1` elements in the response.",
        type=int,
    )
    sub.add_argument(
        "--stop", help="A stop sequence at which to stop generating tokens."
    )
    sub.add_argument(
        "-m",
        "--model",
        required=False,
        help="A model (most commonly a model ID) to generate from. Defaults to the engine's default model.",
    )
    sub.set_defaults(func=Engine.generate)

    # Chat Completions
    sub = subparsers.add_parser("chat_completions.create")

    sub._action_groups.pop()
    req = sub.add_argument_group("required arguments")
    opt = sub.add_argument_group("optional arguments")

    req.add_argument(
        "-m",
        "--model",
        help="The model to use.",
        required=True,
    )
    req.add_argument(
        "-g",
        "--message",
        action="append",
        nargs=2,
        metavar=("ROLE", "CONTENT"),
        help="A message in `{role} {content}` format. Use this argument multiple times to add multiple messages.",
        required=True,
    )
    opt.add_argument(
        "-n",
        "--n",
        help="How many completions to generate for the conversation.",
        type=int,
    )
    opt.add_argument(
        "-M", "--max-tokens", help="The maximum number of tokens to generate.", type=int
    )
    opt.add_argument(
        "-t",
        "--temperature",
        help="""What sampling temperature to use. Higher values means the model will take more risks. Try 0.9 for more creative applications, and 0 (argmax sampling) for ones with a well-defined answer.

Mutually exclusive with `top_p`.""",
        type=float,
    )
    opt.add_argument(
        "-P",
        "--top_p",
        help="""An alternative to sampling with temperature, called nucleus sampling, where the considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10%% probability mass are considered.

            Mutually exclusive with `temperature`.""",
        type=float,
    )
    opt.add_argument(
        "--stop",
        help="A stop sequence at which to stop generating tokens for the message.",
    )
    opt.add_argument(
        "--stream", help="Stream messages as they're ready.", action="store_true"
    )
    sub.set_defaults(func=ChatCompletion.create)

    # Completions
    sub = subparsers.add_parser("completions.create")
    sub.add_argument(
        "-e",
        "--engine",
        help="The engine to use. See https://platform.openai.com/docs/engines for more about what engines are available.",
    )
    sub.add_argument(
        "-m",
        "--model",
        help="The model to use. At most one of `engine` or `model` should be specified.",
    )
    sub.add_argument(
        "--stream", help="Stream tokens as they're ready.", action="store_true"
    )
    sub.add_argument("-p", "--prompt", help="An optional prompt to complete from")
    sub.add_argument(
        "-M", "--max-tokens", help="The maximum number of tokens to generate", type=int
    )
    sub.add_argument(
        "-t",
        "--temperature",
        help="""What sampling temperature to use. Higher values means the model will take more risks. Try 0.9 for more creative applications, and 0 (argmax sampling) for ones with a well-defined answer.

Mutually exclusive with `top_p`.""",
        type=float,
    )
    sub.add_argument(
        "-P",
        "--top_p",
        help="""An alternative to sampling with temperature, called nucleus sampling, where the considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10%% probability mass are considered.

            Mutually exclusive with `temperature`.""",
        type=float,
    )
    sub.add_argument(
        "-n",
        "--n",
        help="How many sub-completions to generate for each prompt.",
        type=int,
    )
    sub.add_argument(
        "--logprobs",
        help="Include the log probabilites on the `logprobs` most likely tokens, as well the chosen tokens. So for example, if `logprobs` is 10, the API will return a list of the 10 most likely tokens. If `logprobs` is 0, only the chosen tokens will have logprobs returned.",
        type=int,
    )
    sub.add_argument(
        "--stop", help="A stop sequence at which to stop generating tokens."
    )
    sub.set_defaults(func=Completion.create)

    # Deployments
    sub = subparsers.add_parser("deployments.list")
    sub.set_defaults(func=Deployment.list)

    sub = subparsers.add_parser("deployments.get")
    sub.add_argument("-i", "--id", required=True, help="The deployment ID")
    sub.set_defaults(func=Deployment.get)

    sub = subparsers.add_parser("deployments.delete")
    sub.add_argument("-i", "--id", required=True, help="The deployment ID")
    sub.set_defaults(func=Deployment.delete)
    
    sub = subparsers.add_parser("deployments.create")
    sub.add_argument("-m", "--model", required=True, help="The model ID")
    sub.add_argument("-s", "--scale_type", required=True, help="The scale type. Either 'manual' or 'standard'")
    sub.set_defaults(func=Deployment.create)

    # Models
    sub = subparsers.add_parser("models.list")
    sub.set_defaults(func=Model.list)

    sub = subparsers.add_parser("models.get")
    sub.add_argument("-i", "--id", required=True, help="The model ID")
    sub.set_defaults(func=Model.get)

    sub = subparsers.add_parser("models.delete")
    sub.add_argument("-i", "--id", required=True, help="The model ID")
    sub.set_defaults(func=Model.delete)

    # Files
    sub = subparsers.add_parser("files.create")

    sub.add_argument(
        "-f",
        "--file",
        required=True,
        help="File to upload",
    )
    sub.add_argument(
        "-p",
        "--purpose",
        help="Why are you uploading this file? (see https://platform.openai.com/docs/api-reference/ for purposes)",
        required=True,
    )
    sub.set_defaults(func=File.create)

    sub = subparsers.add_parser("files.get")
    sub.add_argument("-i", "--id", required=True, help="The files ID")
    sub.set_defaults(func=File.get)

    sub = subparsers.add_parser("files.delete")
    sub.add_argument("-i", "--id", required=True, help="The files ID")
    sub.set_defaults(func=File.delete)

    sub = subparsers.add_parser("files.list")
    sub.set_defaults(func=File.list)

    # Finetune
    sub = subparsers.add_parser("fine_tunes.list")
    sub.set_defaults(func=FineTune.list)

    sub = subparsers.add_parser("fine_tunes.create")
    sub.add_argument(
        "-t",
        "--training_file",
        required=True,
        help="JSONL file containing prompt-completion examples for training. This can "
        "be the ID of a file uploaded through the OpenAI API (e.g. file-abcde12345), "
        'a local file path, or a URL that starts with "http".',
    )
    sub.add_argument(
        "-v",
        "--validation_file",
        help="JSONL file containing prompt-completion examples for validation. This can "
        "be the ID of a file uploaded through the OpenAI API (e.g. file-abcde12345), "
        'a local file path, or a URL that starts with "http".',
    )
    sub.add_argument(
        "--no_check_if_files_exist",
        dest="check_if_files_exist",
        action="store_false",
        help="If this argument is set and training_file or validation_file are file paths, immediately upload them. If this argument is not set, check if they may be duplicates of already uploaded files before uploading, based on file name and file size.",
    )
    sub.add_argument(
        "-m",
        "--model",
        help="The model to start fine-tuning from",
    )
    sub.add_argument(
        "--suffix",
        help="If set, this argument can be used to customize the generated fine-tuned model name."
        "All punctuation and whitespace in `suffix` will be replaced with a "
        "single dash, and the string will be lower cased. The max "
        "length of `suffix` is 40 chars. "
        "The generated name will match the form `{base_model}:ft-{org-title}:{suffix}-{timestamp}`. "
        'For example, `openai api fine_tunes.create -t test.jsonl -m ada --suffix "custom model name" '
        "could generate a model with the name "
        "ada:ft-your-org:custom-model-name-2022-02-15-04-21-04",
    )
    sub.add_argument(
        "--no_follow",
        action="store_true",
        help="If set, returns immediately after creating the job. Otherwise, streams events and waits for the job to complete.",
    )
    sub.add_argument(
        "--n_epochs",
        type=int,
        help="The number of epochs to train the model for. An epoch refers to one "
        "full cycle through the training dataset.",
    )
    sub.add_argument(
        "--batch_size",
        type=int,
        help="The batch size to use for training. The batch size is the number of "
        "training examples used to train a single forward and backward pass.",
    )
    sub.add_argument(
        "--learning_rate_multiplier",
        type=float,
        help="The learning rate multiplier to use for training. The fine-tuning "
        "learning rate is determined by the original learning rate used for "
        "pretraining multiplied by this value.",
    )
    sub.add_argument(
        "--prompt_loss_weight",
        type=float,
        help="The weight to use for the prompt loss. The optimum value here depends "
        "depends on your use case. This determines how much the model prioritizes "
        "learning from prompt tokens vs learning from completion tokens.",
    )
    sub.add_argument(
        "--compute_classification_metrics",
        action="store_true",
        help="If set, we calculate classification-specific metrics such as accuracy "
        "and F-1 score using the validation set at the end of every epoch.",
    )
    sub.set_defaults(compute_classification_metrics=None)
    sub.add_argument(
        "--classification_n_classes",
        type=int,
        help="The number of classes in a classification task. This parameter is "
        "required for multiclass classification.",
    )
    sub.add_argument(
        "--classification_positive_class",
        help="The positive class in binary classification. This parameter is needed "
        "to generate precision, recall and F-1 metrics when doing binary "
        "classification.",
    )
    sub.add_argument(
        "--classification_betas",
        type=float,
        nargs="+",
        help="If this is provided, we calculate F-beta scores at the specified beta "
        "values. The F-beta score is a generalization of F-1 score. This is only "
        "used for binary classification.",
    )
    sub.set_defaults(func=FineTune.create)

    sub = subparsers.add_parser("fine_tunes.get")
    sub.add_argument("-i", "--id", required=True, help="The id of the fine-tune job")
    sub.set_defaults(func=FineTune.get)

    sub = subparsers.add_parser("fine_tunes.results")
    sub.add_argument("-i", "--id", required=True, help="The id of the fine-tune job")
    sub.set_defaults(func=FineTune.results)

    sub = subparsers.add_parser("fine_tunes.events")
    sub.add_argument("-i", "--id", required=True, help="The id of the fine-tune job")

    # TODO(rachel): Remove this in 1.0
    sub.add_argument(
        "-s",
        "--stream",
        action="store_true",
        help="[DEPRECATED] If set, events will be streamed until the job is done. Otherwise, "
        "displays the event history to date.",
    )
    sub.set_defaults(func=FineTune.events)

    sub = subparsers.add_parser("fine_tunes.follow")
    sub.add_argument("-i", "--id", required=True, help="The id of the fine-tune job")
    sub.set_defaults(func=FineTune.follow)

    sub = subparsers.add_parser("fine_tunes.cancel")
    sub.add_argument("-i", "--id", required=True, help="The id of the fine-tune job")
    sub.set_defaults(func=FineTune.cancel)

    sub = subparsers.add_parser("fine_tunes.delete")
    sub.add_argument("-i", "--id", required=True, help="The id of the fine-tune job")
    sub.set_defaults(func=FineTune.delete)

    # Image
    sub = subparsers.add_parser("image.create")
    sub.add_argument("-p", "--prompt", type=str, required=True)
    sub.add_argument("-n", "--num-images", type=int, default=1)
    sub.add_argument(
        "-s", "--size", type=str, default="1024x1024", help="Size of the output image"
    )
    sub.add_argument("--response-format", type=str, default="url")
    sub.set_defaults(func=Image.create)

    sub = subparsers.add_parser("image.create_edit")
    sub.add_argument("-p", "--prompt", type=str, required=True)
    sub.add_argument("-n", "--num-images", type=int, default=1)
    sub.add_argument(
        "-I",
        "--image",
        type=str,
        required=True,
        help="Image to modify. Should be a local path and a PNG encoded image.",
    )
    sub.add_argument(
        "-s", "--size", type=str, default="1024x1024", help="Size of the output image"
    )
    sub.add_argument("--response-format", type=str, default="url")
    sub.add_argument(
        "-M",
        "--mask",
        type=str,
        required=False,
        help="Path to a mask image. It should be the same size as the image you're editing and a RGBA PNG image. The Alpha channel acts as the mask.",
    )
    sub.set_defaults(func=Image.create_edit)

    sub = subparsers.add_parser("image.create_variation")
    sub.add_argument("-n", "--num-images", type=int, default=1)
    sub.add_argument(
        "-I",
        "--image",
        type=str,
        required=True,
        help="Image to modify. Should be a local path and a PNG encoded image.",
    )
    sub.add_argument(
        "-s", "--size", type=str, default="1024x1024", help="Size of the output image"
    )
    sub.add_argument("--response-format", type=str, default="url")
    sub.set_defaults(func=Image.create_variation)

    # Audio
    # transcriptions
    sub = subparsers.add_parser("audio.transcribe")
    # Required
    sub.add_argument("-m", "--model", type=str, default="whisper-1")
    sub.add_argument("-f", "--file", type=str, required=True)
    # Optional
    sub.add_argument("--response-format", type=str)
    sub.add_argument("--language", type=str)
    sub.add_argument("-t", "--temperature", type=float)
    sub.add_argument("--prompt", type=str)
    sub.set_defaults(func=Audio.transcribe)
    # translations
    sub = subparsers.add_parser("audio.translate")
    # Required
    sub.add_argument("-m", "--model", type=str, default="whisper-1")
    sub.add_argument("-f", "--file", type=str, required=True)
    # Optional
    sub.add_argument("--response-format", type=str)
    sub.add_argument("--language", type=str)
    sub.add_argument("-t", "--temperature", type=float)
    sub.add_argument("--prompt", type=str)
    sub.set_defaults(func=Audio.translate)


def wandb_register(parser):
    subparsers = parser.add_subparsers(
        title="wandb", help="Logging with Weights & Biases"
    )

    def help(args):
        parser.print_help()

    parser.set_defaults(func=help)

    sub = subparsers.add_parser("sync")
    sub.add_argument("-i", "--id", help="The id of the fine-tune job (optional)")
    sub.add_argument(
        "-n",
        "--n_fine_tunes",
        type=int,
        default=None,
        help="Number of most recent fine-tunes to log when an id is not provided. By default, every fine-tune is synced.",
    )
    sub.add_argument(
        "--project",
        default="GPT-3",
        help="""Name of the project where you're sending runs. By default, it is "GPT-3".""",
    )
    sub.add_argument(
        "--entity",
        help="Username or team name where you're sending runs. By default, your default entity is used, which is usually your username.",
    )
    sub.add_argument(
        "--force",
        action="store_true",
        help="Forces logging and overwrite existing wandb run of the same fine-tune.",
    )
    sub.set_defaults(force=False)
    sub.set_defaults(func=WandbLogger.sync)
