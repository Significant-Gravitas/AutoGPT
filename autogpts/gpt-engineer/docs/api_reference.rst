.. _api_reference:

=============
API Reference
=============

:mod:`gpt_engineer.cli`: Cli
=============================

.. automodule:: gpt_engineer.cli
    :no-members:
    :no-inherited-members:

Classes
--------------
.. currentmodule:: gpt_engineer

.. autosummary::
    :toctree: cli
    :template: class.rst

    cli.file_selector.DisplayablePath

Functions
--------------
.. currentmodule:: gpt_engineer

.. autosummary::
    :toctree: cli

    cli.collect.collect_learnings
    cli.collect.send_learning
    cli.collect.steps_file_hash
    cli.file_selector.ask_for_files
    cli.file_selector.gui_file_selector
    cli.file_selector.is_in_ignoring_extensions
    cli.file_selector.terminal_file_selector
    cli.learning.ask_if_can_store
    cli.learning.check_consent
    cli.learning.collect_consent
    cli.learning.extract_learning
    cli.learning.get_session
    cli.learning.human_review_input
    cli.learning.logs_to_string
    cli.main.load_env_if_needed
    cli.main.main
    cli.main.preprompts_path

:mod:`gpt_engineer.core`: Core
===============================

.. automodule:: gpt_engineer.core
    :no-members:
    :no-inherited-members:

Classes
--------------
.. currentmodule:: gpt_engineer

.. autosummary::
    :toctree: core
    :template: class.rst

    core.steps.Config

Functions
--------------
.. currentmodule:: gpt_engineer

.. autosummary::
    :toctree: core

    core.ai.create_chat_model
    core.ai.fallback_model
    core.ai.get_tokenizer
    core.ai.serialize_messages
    core.chat_to_files.format_file_to_input
    core.chat_to_files.get_code_strings
    core.chat_to_files.overwrite_files
    core.chat_to_files.parse_chat
    core.chat_to_files.to_files
    core.db.archive
    core.steps.assert_files_ready
    core.steps.clarify
    core.steps.curr_fn
    core.steps.execute_entrypoint
    core.steps.gen_clarified_code
    core.steps.gen_entrypoint
    core.steps.get_improve_prompt
    core.steps.human_review
    core.steps.improve_existing_code
    core.steps.lite_gen
    core.steps.set_improve_filelist
    core.steps.setup_sys_prompt
    core.steps.setup_sys_prompt_existing_code
    core.steps.simple_gen
    core.steps.use_feedback
