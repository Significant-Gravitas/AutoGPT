import json
import random
import commands as cmd
import utils
from memory import get_memory, get_supported_memory_backends
import chat
from colorama import Fore, Style
from spinner import Spinner
import time
import speak
from config import Config
from json_parser import fix_and_parse_json
from ai_config import AIConfig
import traceback
import yaml
import argparse
from logger import logger
import logging
from prompt import get_prompt

cfg = Config()


def check_openai_api_key():
    """检查是否在config.py文件或环境变量中设置了OpenAI API密钥。"""
    if not cfg.openai_api_key:
        print(
            Fore.RED +
            "请在config.py文件或环境变量中设置您的OpenAI API密钥。"
        )
        print("You can get your key from https://beta.openai.com/account/api-keys")
        exit(1)


def attempt_to_fix_json_by_finding_outermost_brackets(json_string):
    if cfg.speak_mode and cfg.debug_mode:
      speak.say_text("我从OpenAI API接收到了无效的JSON响应。现在尝试修复它。")
    logger.typewriter_log("正在尝试通过查找最外层的括号来修复JSON\n")

    try:
        # Use regex to search for JSON objects
        import regex
        json_pattern = regex.compile(r"\{(?:[^{}]|(?R))*\}")
        json_match = json_pattern.search(json_string)

        if json_match:
            # Extract the valid JSON object from the string
            json_string = json_match.group(0)
            logger.typewriter_log(title="显然JSON已经被修复。", title_color=Fore.GREEN)
            if cfg.speak_mode and cfg.debug_mode:
               speak.say_text("显然JSON已经被修复。")
        else:
            raise ValueError("未找到有效的JSON对象")

    except (json.JSONDecodeError, ValueError) as e:
        if cfg.speak_mode:
            speak.say_text("没有工作。我现在不得不忽略这个响应。")
        logger.error("Error:无效的JSON,现在将其设置为空JSON。\n")
        json_string = {}

    return json_string


def print_assistant_thoughts(assistant_reply):
    """将助手的想法打印到控制台。"""
    global ai_name
    global cfg
    try:
        try:
            # Parse and print Assistant response
            assistant_reply_json = fix_and_parse_json(assistant_reply)
        except json.JSONDecodeError as e:
            logger.error("Error: 机器人的想法中包含无效的JSON格式。\n", assistant_reply)
            assistant_reply_json = attempt_to_fix_json_by_finding_outermost_brackets(assistant_reply)
            assistant_reply_json = fix_and_parse_json(assistant_reply_json)

        # Check if assistant_reply_json is a string and attempt to parse it into a JSON object
        if isinstance(assistant_reply_json, str):
            try:
                assistant_reply_json = json.loads(assistant_reply_json)
            except json.JSONDecodeError as e:
                logger.error("Error: 无效的 JSON\n", assistant_reply)
                assistant_reply_json = attempt_to_fix_json_by_finding_outermost_brackets(assistant_reply_json)

        assistant_thoughts_reasoning = None
        assistant_thoughts_plan = None
        assistant_thoughts_speak = None
        assistant_thoughts_criticism = None
        assistant_thoughts = assistant_reply_json.get("thoughts", {})
        assistant_thoughts_text = assistant_thoughts.get("text")

        if assistant_thoughts:
            assistant_thoughts_reasoning = assistant_thoughts.get("reasoning")
            assistant_thoughts_plan = assistant_thoughts.get("plan")
            assistant_thoughts_criticism = assistant_thoughts.get("criticism")
            assistant_thoughts_speak = assistant_thoughts.get("speak")

        logger.typewriter_log(f"{ai_name.upper()} THOUGHTS:", Fore.YELLOW, assistant_thoughts_text)
        logger.typewriter_log("REASONING:", Fore.YELLOW, assistant_thoughts_reasoning)

        if assistant_thoughts_plan:
            logger.typewriter_log("PLAN:", Fore.YELLOW, "")
            # If it's a list, join it into a string
            if isinstance(assistant_thoughts_plan, list):
                assistant_thoughts_plan = "\n".join(assistant_thoughts_plan)
            elif isinstance(assistant_thoughts_plan, dict):
                assistant_thoughts_plan = str(assistant_thoughts_plan)

            # Split the input_string using the newline character and dashes
            lines = assistant_thoughts_plan.split('\n')
            for line in lines:
                line = line.lstrip("- ")
                logger.typewriter_log("- ", Fore.GREEN, line.strip())

        logger.typewriter_log("CRITICISM:", Fore.YELLOW, assistant_thoughts_criticism)
        # Speak the assistant's thoughts
        if cfg.speak_mode and assistant_thoughts_speak:
            speak.say_text(assistant_thoughts_speak)

        return assistant_reply_json
    except json.decoder.JSONDecodeError as e:
        logger.error("Error: 无效的 JSON\n", assistant_reply)
        if cfg.speak_mode:
            speak.say_text("我从OpenAI API接收到了一个无效的JSON响应。我不能忽略这个响应。")

    # All other errors, return "Error: + error message"
    except Exception as e:
        call_stack = traceback.format_exc()
        logger.error("Error: \n", call_stack)


def construct_prompt():
    """生成AI回复的提示文本。"""
    config = AIConfig.load()
    if config.ai_name:
        logger.typewriter_log(
            f"欢迎回来! ",
            Fore.GREEN,
            f"你想让我变回原来的样子吗 {config.ai_name}?",
            speak_text=True)
        should_continue = utils.clean_input(f"""继续上次的这些设置?
姓名:  {config.ai_name}
角色:  {config.ai_role}
目标: {config.ai_goals}
继续 (输入y，继续上一次设置/输入n，重新来过): """)
        if should_continue.lower() == "n":
            config = AIConfig()

    if not config.ai_name:
        config = prompt_user()
        config.save()

    # Get rid of this global:
    global ai_name
    ai_name = config.ai_name

    full_prompt = config.construct_full_prompt()
    return full_prompt


def prompt_user():
    """提示用户输入"""
    ai_name = ""
    # Construct the prompt
    logger.typewriter_log(
        "欢迎来到 Auto-GPT! 中文翻译版由AJ提供, 公众号《阿杰的人生路》获取回复Auto-GPT获取最新版本，加入社区共同探讨使用方式。",
        Fore.GREEN,
        "",
        speak_text=True)


    # Get AI Name from User
    logger.typewriter_log(
        "你的AI机器人名称: ",
        Fore.GREEN,
        "例如, 'AJ-1号-GPT'")
    ai_name = utils.clean_input("AI 机器人名称: ")
    if ai_name == "":
        ai_name = "AJ_NO1_GPTBOT"

    logger.typewriter_log(
        f"{ai_name} 在这里！",
        Fore.LIGHTBLUE_EX,
        "我随时为您服务。",
        speak_text=True)

    # Get AI Role from User
    logger.typewriter_log(
        "描述您的 AI 的角色：",
        Fore.GREEN,
        "例如，'一种旨在自主开发和经营业务的人工智能，其唯一目标是增加你的净资产。")
    ai_role = utils.clean_input(f"{ai_name} 的作用: ")
    if ai_role == "":
        ai_role = "一个旨在自主开发和经营企业以唯一目标增加你净值的人工智能"

    # Enter up to 5 goals for the AI
    logger.typewriter_log(
        "AJ提示你:输入最多5个要AI机器人要帮你实现的功能/目标 ",
        Fore.GREEN,
        "例如：\n增加净值、增加公众号关注者、市场调研、管理多个企业、自主开发网站、App、脚本等等")
    print("输入空白以加载默认值，完成时不要输入任何内容。", flush=True)
    ai_goals = []
    for i in range(5):
        ai_goal = utils.clean_input(f"{Fore.LIGHTBLUE_EX}Goal{Style.RESET_ALL} {i+1}: ")
        if ai_goal == "":
            break
        ai_goals.append(ai_goal)
    if len(ai_goals) == 0:
        ai_goals = ["Increase net worth", "Grow Twitter Account",
                    "Develop and manage multiple businesses autonomously"]

    config = AIConfig(ai_name, ai_role, ai_goals)
    return config


def parse_arguments():
    """解析传递给脚本的参数"""
    global cfg
    cfg.set_debug_mode(False)
    cfg.set_continuous_mode(False)
    cfg.set_speak_mode(False)

    parser = argparse.ArgumentParser(description='Process arguments.')
    parser.add_argument('--continuous', action='store_true', help='Enable Continuous Mode')
    parser.add_argument('--continuous-limit', '-l', type=int, dest="continuous_limit", help='Defines the number of times to run in continuous mode')
    parser.add_argument('--speak', action='store_true', help='Enable Speak Mode')
    parser.add_argument('--debug', action='store_true', help='Enable Debug Mode')
    parser.add_argument('--gpt3only', action='store_true', help='Enable GPT3.5 Only Mode')
    parser.add_argument('--gpt4only', action='store_true', help='Enable GPT4 Only Mode')
    parser.add_argument('--use-memory', '-m', dest="memory_type", help='Defines which Memory backend to use')
    args = parser.parse_args()

    if args.debug:
        logger.typewriter_log("调试模式: ", Fore.GREEN, "启用")
        cfg.set_debug_mode(True)

    if args.continuous:
        logger.typewriter_log("连续模式: ", Fore.RED, "启用")
        logger.typewriter_log("连续模式: ", Fore.RED, "完全由程序控制，直到执行结束，过程无需用户操作，但是可能走偏无法控制，需要注意下它在做什么")
        logger.typewriter_log(
            "警告: ",
            Fore.RED,
            "不推荐连续模式。 它具有潜在危险，可能会导致您的 AI 永远运行或执行您通常不会授权的操作。 使用风险自负。")
        cfg.set_continuous_mode(True)

        if args.continuous_limit:
            logger.typewriter_log(
                "连续模式 限制: ",
                Fore.GREEN,
                f"{args.continuous_limit}")
            cfg.set_continuous_limit(args.continuous_limit)

    # Check if continuous limit is used without continuous mode
    if args.continuous_limit and not args.continuous:
        parser.error("--continuous-limit 只能与--continuous一起使用")

    if args.speak:
        logger.typewriter_log("语音模式: ", Fore.GREEN, "启用")
        cfg.set_speak_mode(True)

    if args.gpt3only:
        logger.typewriter_log("使用 GPT3.5 API: ", Fore.GREEN, "启用")
        cfg.set_smart_llm_model(cfg.fast_llm_model)

    if args.gpt4only:
        logger.typewriter_log("使用 GPT4 API: ", Fore.GREEN, "启用")
        cfg.set_fast_llm_model(cfg.smart_llm_model)

    if args.memory_type:
        supported_memory = get_supported_memory_backends()
        chosen = args.memory_type
        if not chosen in supported_memory:
            logger.typewriter_log("仅支持以下服务去存储内容: ", Fore.RED, f'{supported_memory}')
            logger.typewriter_log(f"默认为: ", Fore.YELLOW, cfg.memory_backend)
        else:
            cfg.memory_backend = chosen


def main():
    global ai_name, memory
    # TODO: fill in llm values here
    check_openai_api_key()
    parse_arguments()
    logger.set_level(logging.DEBUG if cfg.debug_mode else logging.INFO)
    ai_name = ""
    prompt = construct_prompt()
    # print(prompt)
    # Initialize variables
    full_message_history = []
    result = None
    next_action_count = 0
    # Make a constant:
    user_input = "确定要使用的下一个命令，并使用上面指定的格式进行响应:"
    # Initialize memory and make sure it is empty.
    # this is particularly important for indexing and referencing pinecone memory
    memory = get_memory(cfg, init=True)
    print('使用存储的类型: ' + memory.__class__.__name__)
    agent = Agent(
        ai_name=ai_name,
        memory=memory,
        full_message_history=full_message_history,
        next_action_count=next_action_count,
        prompt=prompt,
        user_input=user_input
    )
    agent.start_interaction_loop()


class Agent:
    """与 Auto-GPT 交互的代理类。

     属性：
         ai_name：代理的名称。
         memory：要使用的内存对象。
         full_message_history：完整的消息历史记录。
         next_action_count：要执行的动作数。
         提示：要使用的提示。
         user_input：用户输入。

     """
    def __init__(self,
                 ai_name,
                 memory,
                 full_message_history,
                 next_action_count,
                 prompt,
                 user_input):
        self.ai_name = ai_name
        self.memory = memory
        self.full_message_history = full_message_history
        self.next_action_count = next_action_count
        self.prompt = prompt
        self.user_input = user_input

    def start_interaction_loop(self):
        # Interaction Loop
        loop_count = 0
        while True:
             # Discontinue if continuous limit is reached
            loop_count += 1
            if cfg.continuous_mode and cfg.continuous_limit > 0 and loop_count > cfg.continuous_limit:
                logger.typewriter_log("连续达到限制：", Fore.YELLOW, f"{cfg.continuous_limit}")
                break

            # Send message to AI, get response
            with Spinner("思考... "):
                assistant_reply = chat.chat_with_ai(
                    self.prompt,
                    self.user_input,
                    self.full_message_history,
                    self.memory,
                    cfg.fast_token_limit)  # TODO: This hardcodes the model to use GPT3.5. Make this an argument

            # Print Assistant thoughts
            print_assistant_thoughts(assistant_reply)

            # Get command name and arguments
            try:
                command_name, arguments = cmd.get_command(
                    attempt_to_fix_json_by_finding_outermost_brackets(assistant_reply))
                if cfg.speak_mode:
                    speak.say_text(f"我要执行{command_name}")
            except Exception as e:
                logger.error("Error: \n", str(e))

            if not cfg.continuous_mode and self.next_action_count == 0:
                ### GET USER AUTHORIZATION TO EXECUTE COMMAND ###
                # Get key press: Prompt the user to press enter to continue or escape
                # to exit
                self.user_input = ""
                logger.typewriter_log(
                    "下一步操作: ",
                    Fore.CYAN,
                    f"COMMAND = {Fore.CYAN}{command_name}{Style.RESET_ALL}  ARGUMENTS = {Fore.CYAN}{arguments}{Style.RESET_ALL}")
                print(
                    f"输入'y'授权命令，'y -N'运行N个连续命令，'n'退出程序，或为{ai_name}输入反馈...",
                    flush=True)
                while True:
                    console_input = utils.clean_input(Fore.MAGENTA + "Input:" + Style.RESET_ALL)
                    if console_input.lower().rstrip() == "y":
                        self.user_input = "GENERATE NEXT COMMAND JSON"
                        break
                    elif console_input.lower().startswith("y -"):
                        try:
                            self.next_action_count = abs(int(console_input.split(" ")[1]))
                            self.user_input = "GENERATE NEXT COMMAND JSON"
                        except ValueError:
                            print("输入格式无效。 请输入'y -n',其中 n 是连续任务的数量。 例如: y -1")
                            continue
                        break
                    elif console_input.lower() == "n":
                        self.user_input = "EXIT"
                        break
                    else:
                        self.user_input = console_input
                        command_name = "human_feedback"
                        break

                if self.user_input == "GENERATE NEXT COMMAND JSON":
                    logger.typewriter_log(
                        "-=-=-=-=-=-=-= 用户授权的命令 -=-=-=-=-=-=-=",
                        Fore.MAGENTA,
                        "")
                elif self.user_input == "EXIT":
                    print("退出中...", flush=True)
                    break
            else:
                # Print command
                logger.typewriter_log(
                    "下一步操作: ",
                    Fore.CYAN,
                    f"COMMAND = {Fore.CYAN}{command_name}{Style.RESET_ALL}  ARGUMENTS = {Fore.CYAN}{arguments}{Style.RESET_ALL}")

            # Execute command
            if command_name is not None and command_name.lower().startswith("error"):
                result = f"Command {command_name} 抛出以下错误：" + arguments
            elif command_name == "human_feedback":
                result = f"人工反馈: {self.user_input}"
            else:
                result = f"Command {command_name} returned: {cmd.execute_command(command_name, arguments)}"
                if self.next_action_count > 0:
                    self.next_action_count -= 1

            memory_to_add = f"机器人回复: {assistant_reply} " \
                            f"\结果: {result} " \
                            f"\人工反馈: {self.user_input} "

            self.memory.add(memory_to_add)

            # Check if there's a result from the command append it to the message
            # history
            if result is not None:
                self.full_message_history.append(chat.create_chat_message("system", result))
                logger.typewriter_log("SYSTEM: ", Fore.YELLOW, result)
            else:
                self.full_message_history.append(
                    chat.create_chat_message(
                        "system", "无法执行命令"))
                logger.typewriter_log("SYSTEM: ", Fore.YELLOW, "无法执行命令")


if __name__ == "__main__":
    main()
