import os
import sys
from typing import Any, Callable, NamedTuple, Optional

from openai.datalib import pandas as pd, assert_has_pandas


class Remediation(NamedTuple):
    name: str
    immediate_msg: Optional[str] = None
    necessary_msg: Optional[str] = None
    necessary_fn: Optional[Callable[[Any], Any]] = None
    optional_msg: Optional[str] = None
    optional_fn: Optional[Callable[[Any], Any]] = None
    error_msg: Optional[str] = None


def num_examples_validator(df):
    """
    This validator will only print out the number of examples and recommend to the user to increase the number of examples if less than 100.
    """
    MIN_EXAMPLES = 100
    optional_suggestion = (
        ""
        if len(df) >= MIN_EXAMPLES
        else ". In general, we recommend having at least a few hundred examples. We've found that performance tends to linearly increase for every doubling of the number of examples"
    )
    immediate_msg = (
        f"\n- Your file contains {len(df)} prompt-completion pairs{optional_suggestion}"
    )
    return Remediation(name="num_examples", immediate_msg=immediate_msg)


def necessary_column_validator(df, necessary_column):
    """
    This validator will ensure that the necessary column is present in the dataframe.
    """

    def lower_case_column(df, column):
        cols = [c for c in df.columns if str(c).lower() == column]
        df.rename(columns={cols[0]: column.lower()}, inplace=True)
        return df

    immediate_msg = None
    necessary_fn = None
    necessary_msg = None
    error_msg = None

    if necessary_column not in df.columns:
        if necessary_column in [str(c).lower() for c in df.columns]:

            def lower_case_column_creator(df):
                return lower_case_column(df, necessary_column)

            necessary_fn = lower_case_column_creator
            immediate_msg = (
                f"\n- The `{necessary_column}` column/key should be lowercase"
            )
            necessary_msg = f"Lower case column name to `{necessary_column}`"
        else:
            error_msg = f"`{necessary_column}` column/key is missing. Please make sure you name your columns/keys appropriately, then retry"

    return Remediation(
        name="necessary_column",
        immediate_msg=immediate_msg,
        necessary_msg=necessary_msg,
        necessary_fn=necessary_fn,
        error_msg=error_msg,
    )


def additional_column_validator(df, fields=["prompt", "completion"]):
    """
    This validator will remove additional columns from the dataframe.
    """
    additional_columns = []
    necessary_msg = None
    immediate_msg = None
    necessary_fn = None
    if len(df.columns) > 2:
        additional_columns = [c for c in df.columns if c not in fields]
        warn_message = ""
        for ac in additional_columns:
            dups = [c for c in additional_columns if ac in c]
            if len(dups) > 0:
                warn_message += f"\n  WARNING: Some of the additional columns/keys contain `{ac}` in their name. These will be ignored, and the column/key `{ac}` will be used instead. This could also result from a duplicate column/key in the provided file."
        immediate_msg = f"\n- The input file should contain exactly two columns/keys per row. Additional columns/keys present are: {additional_columns}{warn_message}"
        necessary_msg = f"Remove additional columns/keys: {additional_columns}"

        def necessary_fn(x):
            return x[fields]

    return Remediation(
        name="additional_column",
        immediate_msg=immediate_msg,
        necessary_msg=necessary_msg,
        necessary_fn=necessary_fn,
    )


def non_empty_field_validator(df, field="completion"):
    """
    This validator will ensure that no completion is empty.
    """
    necessary_msg = None
    necessary_fn = None
    immediate_msg = None

    if df[field].apply(lambda x: x == "").any() or df[field].isnull().any():
        empty_rows = (df[field] == "") | (df[field].isnull())
        empty_indexes = df.reset_index().index[empty_rows].tolist()
        immediate_msg = f"\n- `{field}` column/key should not contain empty strings. These are rows: {empty_indexes}"

        def necessary_fn(x):
            return x[x[field] != ""].dropna(subset=[field])

        necessary_msg = f"Remove {len(empty_indexes)} rows with empty {field}s"
    return Remediation(
        name=f"empty_{field}",
        immediate_msg=immediate_msg,
        necessary_msg=necessary_msg,
        necessary_fn=necessary_fn,
    )


def duplicated_rows_validator(df, fields=["prompt", "completion"]):
    """
    This validator will suggest to the user to remove duplicate rows if they exist.
    """
    duplicated_rows = df.duplicated(subset=fields)
    duplicated_indexes = df.reset_index().index[duplicated_rows].tolist()
    immediate_msg = None
    optional_msg = None
    optional_fn = None

    if len(duplicated_indexes) > 0:
        immediate_msg = f"\n- There are {len(duplicated_indexes)} duplicated {'-'.join(fields)} sets. These are rows: {duplicated_indexes}"
        optional_msg = f"Remove {len(duplicated_indexes)} duplicate rows"

        def optional_fn(x):
            return x.drop_duplicates(subset=fields)

    return Remediation(
        name="duplicated_rows",
        immediate_msg=immediate_msg,
        optional_msg=optional_msg,
        optional_fn=optional_fn,
    )


def long_examples_validator(df):
    """
    This validator will suggest to the user to remove examples that are too long.
    """
    immediate_msg = None
    optional_msg = None
    optional_fn = None

    ft_type = infer_task_type(df)
    if ft_type != "open-ended generation":
        def get_long_indexes(d):
            long_examples = d.apply(
                lambda x: len(x.prompt) + len(x.completion) > 10000, axis=1
            )
            return d.reset_index().index[long_examples].tolist()

        long_indexes = get_long_indexes(df)

        if len(long_indexes) > 0:
            immediate_msg = f"\n- There are {len(long_indexes)} examples that are very long. These are rows: {long_indexes}\nFor conditional generation, and for classification the examples shouldn't be longer than 2048 tokens."
            optional_msg = f"Remove {len(long_indexes)} long examples"

            def optional_fn(x):
                
                long_indexes_to_drop = get_long_indexes(x)
                if long_indexes != long_indexes_to_drop:
                    sys.stdout.write(f"The indices of the long examples has changed as a result of a previously applied recommendation.\nThe {len(long_indexes_to_drop)} long examples to be dropped are now at the following indices: {long_indexes_to_drop}\n")
                return x.drop(long_indexes_to_drop)

    return Remediation(
        name="long_examples",
        immediate_msg=immediate_msg,
        optional_msg=optional_msg,
        optional_fn=optional_fn,
    )


def common_prompt_suffix_validator(df):
    """
    This validator will suggest to add a common suffix to the prompt if one doesn't already exist in case of classification or conditional generation.
    """
    error_msg = None
    immediate_msg = None
    optional_msg = None
    optional_fn = None

    # Find a suffix which is not contained within the prompt otherwise
    suggested_suffix = "\n\n### =>\n\n"
    suffix_options = [
        " ->",
        "\n\n###\n\n",
        "\n\n===\n\n",
        "\n\n---\n\n",
        "\n\n===>\n\n",
        "\n\n--->\n\n",
    ]
    for suffix_option in suffix_options:
        if suffix_option == " ->":
            if df.prompt.str.contains("\n").any():
                continue
        if df.prompt.str.contains(suffix_option, regex=False).any():
            continue
        suggested_suffix = suffix_option
        break
    display_suggested_suffix = suggested_suffix.replace("\n", "\\n")

    ft_type = infer_task_type(df)
    if ft_type == "open-ended generation":
        return Remediation(name="common_suffix")

    def add_suffix(x, suffix):
        x["prompt"] += suffix
        return x

    common_suffix = get_common_xfix(df.prompt, xfix="suffix")
    if (df.prompt == common_suffix).all():
        error_msg = f"All prompts are identical: `{common_suffix}`\nConsider leaving the prompts blank if you want to do open-ended generation, otherwise ensure prompts are different"
        return Remediation(name="common_suffix", error_msg=error_msg)

    if common_suffix != "":
        common_suffix_new_line_handled = common_suffix.replace("\n", "\\n")
        immediate_msg = (
            f"\n- All prompts end with suffix `{common_suffix_new_line_handled}`"
        )
        if len(common_suffix) > 10:
            immediate_msg += f". This suffix seems very long. Consider replacing with a shorter suffix, such as `{display_suggested_suffix}`"
        if (
            df.prompt.str[: -len(common_suffix)]
            .str.contains(common_suffix, regex=False)
            .any()
        ):
            immediate_msg += f"\n  WARNING: Some of your prompts contain the suffix `{common_suffix}` more than once. We strongly suggest that you review your prompts and add a unique suffix"

    else:
        immediate_msg = "\n- Your data does not contain a common separator at the end of your prompts. Having a separator string appended to the end of the prompt makes it clearer to the fine-tuned model where the completion should begin. See https://platform.openai.com/docs/guides/fine-tuning/preparing-your-dataset for more detail and examples. If you intend to do open-ended generation, then you should leave the prompts empty"

    if common_suffix == "":
        optional_msg = (
            f"Add a suffix separator `{display_suggested_suffix}` to all prompts"
        )

        def optional_fn(x):
            return add_suffix(x, suggested_suffix)

    return Remediation(
        name="common_completion_suffix",
        immediate_msg=immediate_msg,
        optional_msg=optional_msg,
        optional_fn=optional_fn,
        error_msg=error_msg,
    )


def common_prompt_prefix_validator(df):
    """
    This validator will suggest to remove a common prefix from the prompt if a long one exist.
    """
    MAX_PREFIX_LEN = 12

    immediate_msg = None
    optional_msg = None
    optional_fn = None

    common_prefix = get_common_xfix(df.prompt, xfix="prefix")
    if common_prefix == "":
        return Remediation(name="common_prefix")

    def remove_common_prefix(x, prefix):
        x["prompt"] = x["prompt"].str[len(prefix) :]
        return x

    if (df.prompt == common_prefix).all():
        # already handled by common_suffix_validator
        return Remediation(name="common_prefix")

    if common_prefix != "":
        immediate_msg = f"\n- All prompts start with prefix `{common_prefix}`"
        if MAX_PREFIX_LEN < len(common_prefix):
            immediate_msg += ". Fine-tuning doesn't require the instruction specifying the task, or a few-shot example scenario. Most of the time you should only add the input data into the prompt, and the desired output into the completion"
            optional_msg = f"Remove prefix `{common_prefix}` from all prompts"

            def optional_fn(x):
                return remove_common_prefix(x, common_prefix)

    return Remediation(
        name="common_prompt_prefix",
        immediate_msg=immediate_msg,
        optional_msg=optional_msg,
        optional_fn=optional_fn,
    )


def common_completion_prefix_validator(df):
    """
    This validator will suggest to remove a common prefix from the completion if a long one exist.
    """
    MAX_PREFIX_LEN = 5

    common_prefix = get_common_xfix(df.completion, xfix="prefix")
    ws_prefix = len(common_prefix) > 0 and common_prefix[0] == " "
    if len(common_prefix) < MAX_PREFIX_LEN:
        return Remediation(name="common_prefix")

    def remove_common_prefix(x, prefix, ws_prefix):
        x["completion"] = x["completion"].str[len(prefix) :]
        if ws_prefix:
            # keep the single whitespace as prefix
            x["completion"] = " " + x["completion"]
        return x

    if (df.completion == common_prefix).all():
        # already handled by common_suffix_validator
        return Remediation(name="common_prefix")

    immediate_msg = f"\n- All completions start with prefix `{common_prefix}`. Most of the time you should only add the output data into the completion, without any prefix"
    optional_msg = f"Remove prefix `{common_prefix}` from all completions"

    def optional_fn(x):
        return remove_common_prefix(x, common_prefix, ws_prefix)

    return Remediation(
        name="common_completion_prefix",
        immediate_msg=immediate_msg,
        optional_msg=optional_msg,
        optional_fn=optional_fn,
    )


def common_completion_suffix_validator(df):
    """
    This validator will suggest to add a common suffix to the completion if one doesn't already exist in case of classification or conditional generation.
    """
    error_msg = None
    immediate_msg = None
    optional_msg = None
    optional_fn = None

    ft_type = infer_task_type(df)
    if ft_type == "open-ended generation" or ft_type == "classification":
        return Remediation(name="common_suffix")

    common_suffix = get_common_xfix(df.completion, xfix="suffix")
    if (df.completion == common_suffix).all():
        error_msg = f"All completions are identical: `{common_suffix}`\nEnsure completions are different, otherwise the model will just repeat `{common_suffix}`"
        return Remediation(name="common_suffix", error_msg=error_msg)

    # Find a suffix which is not contained within the completion otherwise
    suggested_suffix = " [END]"
    suffix_options = [
        "\n",
        ".",
        " END",
        "***",
        "+++",
        "&&&",
        "$$$",
        "@@@",
        "%%%",
    ]
    for suffix_option in suffix_options:
        if df.completion.str.contains(suffix_option, regex=False).any():
            continue
        suggested_suffix = suffix_option
        break
    display_suggested_suffix = suggested_suffix.replace("\n", "\\n")

    def add_suffix(x, suffix):
        x["completion"] += suffix
        return x

    if common_suffix != "":
        common_suffix_new_line_handled = common_suffix.replace("\n", "\\n")
        immediate_msg = (
            f"\n- All completions end with suffix `{common_suffix_new_line_handled}`"
        )
        if len(common_suffix) > 10:
            immediate_msg += f". This suffix seems very long. Consider replacing with a shorter suffix, such as `{display_suggested_suffix}`"
        if (
            df.completion.str[: -len(common_suffix)]
            .str.contains(common_suffix, regex=False)
            .any()
        ):
            immediate_msg += f"\n  WARNING: Some of your completions contain the suffix `{common_suffix}` more than once. We suggest that you review your completions and add a unique ending"

    else:
        immediate_msg = "\n- Your data does not contain a common ending at the end of your completions. Having a common ending string appended to the end of the completion makes it clearer to the fine-tuned model where the completion should end. See https://platform.openai.com/docs/guides/fine-tuning/preparing-your-dataset for more detail and examples."

    if common_suffix == "":
        optional_msg = (
            f"Add a suffix ending `{display_suggested_suffix}` to all completions"
        )

        def optional_fn(x):
            return add_suffix(x, suggested_suffix)

    return Remediation(
        name="common_completion_suffix",
        immediate_msg=immediate_msg,
        optional_msg=optional_msg,
        optional_fn=optional_fn,
        error_msg=error_msg,
    )


def completions_space_start_validator(df):
    """
    This validator will suggest to add a space at the start of the completion if it doesn't already exist. This helps with tokenization.
    """

    def add_space_start(x):
        x["completion"] = x["completion"].apply(
            lambda x: ("" if x[0] == " " else " ") + x
        )
        return x

    optional_msg = None
    optional_fn = None
    immediate_msg = None

    if df.completion.str[:1].nunique() != 1 or df.completion.values[0][0] != " ":
        immediate_msg = "\n- The completion should start with a whitespace character (` `). This tends to produce better results due to the tokenization we use. See https://platform.openai.com/docs/guides/fine-tuning/preparing-your-dataset for more details"
        optional_msg = "Add a whitespace character to the beginning of the completion"
        optional_fn = add_space_start
    return Remediation(
        name="completion_space_start",
        immediate_msg=immediate_msg,
        optional_msg=optional_msg,
        optional_fn=optional_fn,
    )


def lower_case_validator(df, column):
    """
    This validator will suggest to lowercase the column values, if more than a third of letters are uppercase.
    """

    def lower_case(x):
        x[column] = x[column].str.lower()
        return x

    count_upper = (
        df[column]
        .apply(lambda x: sum(1 for c in x if c.isalpha() and c.isupper()))
        .sum()
    )
    count_lower = (
        df[column]
        .apply(lambda x: sum(1 for c in x if c.isalpha() and c.islower()))
        .sum()
    )

    if count_upper * 2 > count_lower:
        return Remediation(
            name="lower_case",
            immediate_msg=f"\n- More than a third of your `{column}` column/key is uppercase. Uppercase {column}s tends to perform worse than a mixture of case encountered in normal language. We recommend to lower case the data if that makes sense in your domain. See https://platform.openai.com/docs/guides/fine-tuning/preparing-your-dataset for more details",
            optional_msg=f"Lowercase all your data in column/key `{column}`",
            optional_fn=lower_case,
        )


def read_any_format(fname, fields=["prompt", "completion"]):
    """
    This function will read a file saved in .csv, .json, .txt, .xlsx or .tsv format using pandas.
     - for .xlsx it will read the first sheet
     - for .txt it will assume completions and split on newline
    """
    assert_has_pandas()
    remediation = None
    necessary_msg = None
    immediate_msg = None
    error_msg = None
    df = None

    if os.path.isfile(fname):
        try:
            if fname.lower().endswith(".csv") or fname.lower().endswith(".tsv"):
                file_extension_str, separator = (
                    ("CSV", ",") if fname.lower().endswith(".csv") else ("TSV", "\t")
                )
                immediate_msg = f"\n- Based on your file extension, your file is formatted as a {file_extension_str} file"
                necessary_msg = (
                    f"Your format `{file_extension_str}` will be converted to `JSONL`"
                )
                df = pd.read_csv(fname, sep=separator, dtype=str).fillna("")
            elif fname.lower().endswith(".xlsx"):
                immediate_msg = "\n- Based on your file extension, your file is formatted as an Excel file"
                necessary_msg = "Your format `XLSX` will be converted to `JSONL`"
                xls = pd.ExcelFile(fname)
                sheets = xls.sheet_names
                if len(sheets) > 1:
                    immediate_msg += "\n- Your Excel file contains more than one sheet. Please either save as csv or ensure all data is present in the first sheet. WARNING: Reading only the first sheet..."
                df = pd.read_excel(fname, dtype=str).fillna("")
            elif fname.lower().endswith(".txt"):
                immediate_msg = (
                    "\n- Based on your file extension, you provided a text file"
                )
                necessary_msg = "Your format `TXT` will be converted to `JSONL`"
                with open(fname, "r") as f:
                    content = f.read()
                    df = pd.DataFrame(
                        [["", line] for line in content.split("\n")],
                        columns=fields,
                        dtype=str,
                    ).fillna("")
            elif fname.lower().endswith(".jsonl"):
                df = pd.read_json(fname, lines=True, dtype=str).fillna("")
                if len(df) == 1:
                    # this is NOT what we expect for a .jsonl file
                    immediate_msg = "\n- Your JSONL file appears to be in a JSON format. Your file will be converted to JSONL format"
                    necessary_msg = "Your format `JSON` will be converted to `JSONL`"
                    df = pd.read_json(fname, dtype=str).fillna("")
                else:
                    pass  # this is what we expect for a .jsonl file
            elif fname.lower().endswith(".json"):
                df = pd.read_json(fname, lines=True, dtype=str).fillna("")
                if len(df) == 1:
                    # this is what we expect for a .json file
                    df = pd.read_json(fname, dtype=str).fillna("")
                else:
                    # this is NOT what we expect for a .json file
                    immediate_msg = "\n- Your JSON file appears to be in a JSONL format. Your file will be converted to JSONL format"
                    necessary_msg = "Your format `JSON` will be converted to `JSONL`"
            else:
                error_msg = "Your file must have one of the following extensions: .CSV, .TSV, .XLSX, .TXT, .JSON or .JSONL"
                if "." in fname:
                    error_msg += f" Your file `{fname}` ends with the extension `.{fname.split('.')[-1]}` which is not supported."
                else:
                    error_msg += f" Your file `{fname}` is missing a file extension."

        except (ValueError, TypeError):
            file_extension_str = fname.split(".")[-1].upper()
            error_msg = f"Your file `{fname}` does not appear to be in valid {file_extension_str} format. Please ensure your file is formatted as a valid {file_extension_str} file."

    else:
        error_msg = f"File {fname} does not exist."

    remediation = Remediation(
        name="read_any_format",
        necessary_msg=necessary_msg,
        immediate_msg=immediate_msg,
        error_msg=error_msg,
    )
    return df, remediation


def format_inferrer_validator(df):
    """
    This validator will infer the likely fine-tuning format of the data, and display it to the user if it is classification.
    It will also suggest to use ada and explain train/validation split benefits.
    """
    ft_type = infer_task_type(df)
    immediate_msg = None
    if ft_type == "classification":
        immediate_msg = f"\n- Based on your data it seems like you're trying to fine-tune a model for {ft_type}\n- For classification, we recommend you try one of the faster and cheaper models, such as `ada`\n- For classification, you can estimate the expected model performance by keeping a held out dataset, which is not used for training"
    return Remediation(name="num_examples", immediate_msg=immediate_msg)


def apply_necessary_remediation(df, remediation):
    """
    This function will apply a necessary remediation to a dataframe, or print an error message if one exists.
    """
    if remediation.error_msg is not None:
        sys.stderr.write(
            f"\n\nERROR in {remediation.name} validator: {remediation.error_msg}\n\nAborting..."
        )
        sys.exit(1)
    if remediation.immediate_msg is not None:
        sys.stdout.write(remediation.immediate_msg)
    if remediation.necessary_fn is not None:
        df = remediation.necessary_fn(df)
    return df


def accept_suggestion(input_text, auto_accept):
    sys.stdout.write(input_text)
    if auto_accept:
        sys.stdout.write("Y\n")
        return True
    return input().lower() != "n"


def apply_optional_remediation(df, remediation, auto_accept):
    """
    This function will apply an optional remediation to a dataframe, based on the user input.
    """
    optional_applied = False
    input_text = f"- [Recommended] {remediation.optional_msg} [Y/n]: "
    if remediation.optional_msg is not None:
        if accept_suggestion(input_text, auto_accept):
            df = remediation.optional_fn(df)
            optional_applied = True
    if remediation.necessary_msg is not None:
        sys.stdout.write(f"- [Necessary] {remediation.necessary_msg}\n")
    return df, optional_applied


def estimate_fine_tuning_time(df):
    """
    Estimate the time it'll take to fine-tune the dataset
    """
    ft_format = infer_task_type(df)
    expected_time = 1.0
    if ft_format == "classification":
        num_examples = len(df)
        expected_time = num_examples * 1.44
    else:
        size = df.memory_usage(index=True).sum()
        expected_time = size * 0.0515

    def format_time(time):
        if time < 60:
            return f"{round(time, 2)} seconds"
        elif time < 3600:
            return f"{round(time / 60, 2)} minutes"
        elif time < 86400:
            return f"{round(time / 3600, 2)} hours"
        else:
            return f"{round(time / 86400, 2)} days"

    time_string = format_time(expected_time + 140)
    sys.stdout.write(
        f"Once your model starts training, it'll approximately take {time_string} to train a `curie` model, and less for `ada` and `babbage`. Queue will approximately take half an hour per job ahead of you.\n"
    )


def get_outfnames(fname, split):
    suffixes = ["_train", "_valid"] if split else [""]
    i = 0
    while True:
        index_suffix = f" ({i})" if i > 0 else ""
        candidate_fnames = [
            os.path.splitext(fname)[0] + "_prepared" + suffix + index_suffix + ".jsonl"
            for suffix in suffixes
        ]
        if not any(os.path.isfile(f) for f in candidate_fnames):
            return candidate_fnames
        i += 1


def get_classification_hyperparams(df):
    n_classes = df.completion.nunique()
    pos_class = None
    if n_classes == 2:
        pos_class = df.completion.value_counts().index[0]
    return n_classes, pos_class


def write_out_file(df, fname, any_remediations, auto_accept):
    """
    This function will write out a dataframe to a file, if the user would like to proceed, and also offer a fine-tuning command with the newly created file.
    For classification it will optionally ask the user if they would like to split the data into train/valid files, and modify the suggested command to include the valid set.
    """
    ft_format = infer_task_type(df)
    common_prompt_suffix = get_common_xfix(df.prompt, xfix="suffix")
    common_completion_suffix = get_common_xfix(df.completion, xfix="suffix")

    split = False
    input_text = "- [Recommended] Would you like to split into training and validation set? [Y/n]: "
    if ft_format == "classification":
        if accept_suggestion(input_text, auto_accept):
            split = True

    additional_params = ""
    common_prompt_suffix_new_line_handled = common_prompt_suffix.replace("\n", "\\n")
    common_completion_suffix_new_line_handled = common_completion_suffix.replace(
        "\n", "\\n"
    )
    optional_ending_string = (
        f' Make sure to include `stop=["{common_completion_suffix_new_line_handled}"]` so that the generated texts ends at the expected place.'
        if len(common_completion_suffix_new_line_handled) > 0
        else ""
    )

    input_text = "\n\nYour data will be written to a new JSONL file. Proceed [Y/n]: "

    if not any_remediations and not split:
        sys.stdout.write(
            f'\nYou can use your file for fine-tuning:\n> openai api fine_tunes.create -t "{fname}"{additional_params}\n\nAfter you’ve fine-tuned a model, remember that your prompt has to end with the indicator string `{common_prompt_suffix_new_line_handled}` for the model to start generating completions, rather than continuing with the prompt.{optional_ending_string}\n'
        )
        estimate_fine_tuning_time(df)

    elif accept_suggestion(input_text, auto_accept):
        fnames = get_outfnames(fname, split)
        if split:
            assert len(fnames) == 2 and "train" in fnames[0] and "valid" in fnames[1]
            MAX_VALID_EXAMPLES = 1000
            n_train = max(len(df) - MAX_VALID_EXAMPLES, int(len(df) * 0.8))
            df_train = df.sample(n=n_train, random_state=42)
            df_valid = df.drop(df_train.index)
            df_train[["prompt", "completion"]].to_json(
                fnames[0], lines=True, orient="records", force_ascii=False
            )
            df_valid[["prompt", "completion"]].to_json(
                fnames[1], lines=True, orient="records", force_ascii=False
            )

            n_classes, pos_class = get_classification_hyperparams(df)
            additional_params += " --compute_classification_metrics"
            if n_classes == 2:
                additional_params += f' --classification_positive_class "{pos_class}"'
            else:
                additional_params += f" --classification_n_classes {n_classes}"
        else:
            assert len(fnames) == 1
            df[["prompt", "completion"]].to_json(
                fnames[0], lines=True, orient="records", force_ascii=False
            )

        # Add -v VALID_FILE if we split the file into train / valid
        files_string = ("s" if split else "") + " to `" + ("` and `".join(fnames))
        valid_string = f' -v "{fnames[1]}"' if split else ""
        separator_reminder = (
            ""
            if len(common_prompt_suffix_new_line_handled) == 0
            else f"After you’ve fine-tuned a model, remember that your prompt has to end with the indicator string `{common_prompt_suffix_new_line_handled}` for the model to start generating completions, rather than continuing with the prompt."
        )
        sys.stdout.write(
            f'\nWrote modified file{files_string}`\nFeel free to take a look!\n\nNow use that file when fine-tuning:\n> openai api fine_tunes.create -t "{fnames[0]}"{valid_string}{additional_params}\n\n{separator_reminder}{optional_ending_string}\n'
        )
        estimate_fine_tuning_time(df)
    else:
        sys.stdout.write("Aborting... did not write the file\n")


def infer_task_type(df):
    """
    Infer the likely fine-tuning task type from the data
    """
    CLASSIFICATION_THRESHOLD = 3  # min_average instances of each class
    if sum(df.prompt.str.len()) == 0:
        return "open-ended generation"

    if len(df.completion.unique()) < len(df) / CLASSIFICATION_THRESHOLD:
        return "classification"

    return "conditional generation"


def get_common_xfix(series, xfix="suffix"):
    """
    Finds the longest common suffix or prefix of all the values in a series
    """
    common_xfix = ""
    while True:
        common_xfixes = (
            series.str[-(len(common_xfix) + 1) :]
            if xfix == "suffix"
            else series.str[: len(common_xfix) + 1]
        )  # first few or last few characters
        if (
            common_xfixes.nunique() != 1
        ):  # we found the character at which we don't have a unique xfix anymore
            break
        elif (
            common_xfix == common_xfixes.values[0]
        ):  # the entire first row is a prefix of every other row
            break
        else:  # the first or last few characters are still common across all rows - let's try to add one more
            common_xfix = common_xfixes.values[0]
    return common_xfix


def get_validators():
    return [
        num_examples_validator,
        lambda x: necessary_column_validator(x, "prompt"),
        lambda x: necessary_column_validator(x, "completion"),
        additional_column_validator,
        non_empty_field_validator,
        format_inferrer_validator,
        duplicated_rows_validator,
        long_examples_validator,
        lambda x: lower_case_validator(x, "prompt"),
        lambda x: lower_case_validator(x, "completion"),
        common_prompt_suffix_validator,
        common_prompt_prefix_validator,
        common_completion_prefix_validator,
        common_completion_suffix_validator,
        completions_space_start_validator,
    ]


def apply_validators(
    df,
    fname,
    remediation,
    validators,
    auto_accept,
    write_out_file_func,
):
    optional_remediations = []
    if remediation is not None:
        optional_remediations.append(remediation)
    for validator in validators:
        remediation = validator(df)
        if remediation is not None:
            optional_remediations.append(remediation)
            df = apply_necessary_remediation(df, remediation)

    any_optional_or_necessary_remediations = any(
        [
            remediation
            for remediation in optional_remediations
            if remediation.optional_msg is not None
            or remediation.necessary_msg is not None
        ]
    )
    any_necessary_applied = any(
        [
            remediation
            for remediation in optional_remediations
            if remediation.necessary_msg is not None
        ]
    )
    any_optional_applied = False

    if any_optional_or_necessary_remediations:
        sys.stdout.write(
            "\n\nBased on the analysis we will perform the following actions:\n"
        )
        for remediation in optional_remediations:
            df, optional_applied = apply_optional_remediation(
                df, remediation, auto_accept
            )
            any_optional_applied = any_optional_applied or optional_applied
    else:
        sys.stdout.write("\n\nNo remediations found.\n")

    any_optional_or_necessary_applied = any_optional_applied or any_necessary_applied

    write_out_file_func(df, fname, any_optional_or_necessary_applied, auto_accept)
