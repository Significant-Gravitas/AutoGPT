import unittest

from autogpt.agent.agent import should_prompt_user, generate_user_feedback_message, \
    calculate_next_command_from_user_input


class TestAgent(unittest.TestCase):
    def test_should_prompt_user_given_not_in_continuous_mode_and_next_action_count_is_zero_returns_true(self):
        self.assertTrue(should_prompt_user(False, 0))

    def test_should_prompt_user_given_not_in_continuous_mode_and_next_action_count_is_greater_than_zero_returns_false(
            self):
        self.assertFalse(should_prompt_user(False, 2))

    def test_should_prompt_user_given_in_continuous_mode_and_next_action_count_is_zero_returns_false(self):
        self.assertFalse(should_prompt_user(True, 0))

    def test_should_prompt_user_given_in_continuous_mode_and_next_action_count_is_greater_than_zero_returns_false(self):
        self.assertFalse(should_prompt_user(True, 100))

    def test_generate_user_feedback_message_should_format_message_with_ai_name(self):
        expected_message = "Enter 'y' to authorise command, 'y -N' to run N continuous commands," \
                           "'n' to exit program, or enter feedback for the_ai_name... "
        self.assertEquals(generate_user_feedback_message("the_ai_name"), expected_message)

    def test_calculate_next_command_from_user_input_given_user_entered_y_result_is_generate_next_command(self):
        command_name, next_count, user_input = calculate_next_command_from_user_input("y")
        self.assertIsNone(command_name)
        self.assertEquals(next_count, 0)
        self.assertEquals(user_input, "GENERATE NEXT COMMAND JSON")

    def test_calculate_next_command_from_user_input_given_user_entered_y_dash_number_result_generate_next_command(self):
        command_name, next_count, user_input = calculate_next_command_from_user_input("y -20")
        self.assertIsNone(command_name)
        self.assertEquals(next_count, 20)
        self.assertEquals(user_input, "GENERATE NEXT COMMAND JSON")

    def test_calculate_next_command_from_user_input_given_user_entered_y_dash_and_invalid_number_print_error(self):
        command_name, next_count, user_input = calculate_next_command_from_user_input("y -X")
        self.assertEquals(command_name, "input error")
        self.assertEquals(next_count, 0)
        self.assertEquals(user_input, "Invalid input format. Please enter 'y -n' where n is the number of continuous "
                                      "tasks.")

    def test_calculate_next_command_from_user_input_given_user_entered_n_result_is_exit(self):
        command_name, next_count, user_input = calculate_next_command_from_user_input("n")
        self.assertIsNone(command_name)
        self.assertEquals(next_count, 0)
        self.assertEquals(user_input, "EXIT")

    def test_calculate_next_command_from_user_input_given_user_text_result_is_feedback(self):
        the_input = "Some item to tell the AI about while it is doing stuff."
        command_name, next_count, user_input = calculate_next_command_from_user_input(the_input)
        self.assertEquals(command_name, "human_feedback")
        self.assertEquals(next_count, 0)
        self.assertEquals(user_input, the_input)
