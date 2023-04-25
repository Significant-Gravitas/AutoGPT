from autogpt.agent.agent import should_prompt_user, generate_user_feedback_message, \
    calculate_next_command_from_user_input


def test_should_prompt_user_given_not_in_continuous_mode_and_next_action_count_is_zero_returns_true():
    assert should_prompt_user(False, 0) is True


def test_should_prompt_user_given_not_in_continuous_mode_and_next_action_count_is_greater_than_zero_returns_false():
    assert should_prompt_user(False, 2) is False


def test_should_prompt_user_given_in_continuous_mode_and_next_action_count_is_zero_returns_false():
    assert should_prompt_user(True, 0) is False


def test_should_prompt_user_given_in_continuous_mode_and_next_action_count_is_greater_than_zero_returns_false():
    assert should_prompt_user(True, 100) is False


def test_generate_user_feedback_message_should_format_message_with_ai_name():
    expected_message = "Enter 'y' to authorise command, 'y -N' to run N continuous commands, " \
                       "'s' to run self-feedback commands, " \
                       "'n' to exit program, or enter feedback for the_ai_name... "
    assert generate_user_feedback_message("the_ai_name") == expected_message


def test_calculate_next_command_from_user_input_given_user_entered_y_result_is_generate_next_command():
    command_name, next_count, user_input = calculate_next_command_from_user_input("y")
    assert command_name is None
    assert next_count == 0
    assert user_input == "GENERATE NEXT COMMAND JSON"


def test_calculate_next_command_from_user_input_given_user_entered_y_dash_number_result_generate_next_command():
    command_name, next_count, user_input = calculate_next_command_from_user_input("y -20")
    assert command_name is None
    assert next_count == 20
    assert user_input == "GENERATE NEXT COMMAND JSON"


def test_calculate_next_command_from_user_input_given_user_entered_y_dash_and_invalid_number_print_error():
    command_name, next_count, user_input = calculate_next_command_from_user_input("y -X")
    assert command_name == "input error"
    assert next_count == 0
    assert user_input == "Invalid input format. Please enter 'y -n' where n is the number of continuous tasks."


def test_calculate_next_command_from_user_input_given_user_entered_n_result_is_exit():
    command_name, next_count, user_input = calculate_next_command_from_user_input("n")
    assert command_name is None
    assert next_count == 0
    assert user_input == "EXIT"


def test_calculate_next_command_from_user_input_given_user_text_result_is_feedback():
    the_input = "Some item to tell the AI about while it is doing stuff."
    command_name, next_count, user_input = calculate_next_command_from_user_input(the_input)
    assert command_name == "human_feedback"
    assert next_count == 0
    assert user_input == the_input


def test_calculate_next_command_from_user_input_given_s_turn_on_verify_feedback():
    the_input = "s"
    command_name, next_count, user_input = calculate_next_command_from_user_input(the_input)
    assert command_name == "self_feedback"
    assert next_count == 0
    assert user_input is None


def test_calculate_next_command_from_user_input_given_no_value_print_error():
    the_input = "      "
    command_name, next_count, user_input = calculate_next_command_from_user_input(the_input)
    assert command_name == "input error"
    assert next_count == 0
    assert user_input == "Invalid input format."
