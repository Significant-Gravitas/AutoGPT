import lmql


# @lmql.query
# async def smart_select_agent(message):
#     '''
#     argmax
#         pass
#     from
#         "openai/text-davinci-003
#     '''


@lmql.query
async def classify_emotion(message):
    '''
    argmax
        """Review:{message}\n
        Q: What is the underlying sentiment of this review and why?\n
        A:[ANALYSIS]\n
        Based on this, the overall sentiment of the message can be considered to be[CLASSIFICATION]"""
    from
        "openai/text-davinci-003"
    distribution
        CLASSIFICATION in [" agreement", " critique", " surprise", " annoyance", " neutral", " amusement", " idea", " sad"]
    '''


@lmql.query
async def create_chat_completion_gpt4(messages):
    '''
    argmax
        for message in messages:
            if message['role'] == 'system':
                "{:system} {message['content']}"
            elif message['role'] == 'user':
                "{:user} {message['content']}"
            elif message['role'] == 'assistant':
                "{:assistant} {message['content']}"
            else:
                assert False, "not a supported role " + str(role)
        schema = {
            "thoughts": {
                "text": str,
                "reasoning": str,
                "plan": str,
                "criticism": str,
                "speak": str
            },
            "command": {
                "name": str,
                "args": {
                    "[STRING_VALUE]" : str
                }
            }
        }
        stack = [("", schema)]
        indent = ""
        dq = '"'
        while len(stack) > 0:
            t = stack.pop(0)
            if type(t) is tuple:
                k, key_type = t
                if k != "":
                    "{indent}{dq}{k}{dq}: "
                if key_type is str:
                     if stack[0] == "DEDENT":
                        '"[STRING_VALUE]\n'
                     else:
                        '"[STRING_VALUE],\n'
                elif key_type is int:
                     if stack[0] == "DEDENT":
                        "[INT_VALUE]\n"
                     else:
                        "[INT_VALUE],\n"
                elif type(key_type) is dict:
                    "{{\n"
                    indent += "    "
                    if len(stack) == 0 or stack[0] == "DEDENT":
                        stack = [(k, key_type[k]) for k in key_type.keys()] + ["DEDENT", "}\n"] + stack
                    else:
                        stack = [(k, key_type[k]) for k in key_type.keys()] + ["DEDENT", "},\n"] + stack
                else:
                    assert False, "not a supported type " + str(k)
            elif t == "DEDENT":
                indent = indent[:-4]
            elif type(t) is str:
                "{indent}{t}"
            else:
                assert False, "not a supported type" + str(t)
    from
       'openai/gpt-4'
    where
       STOPS_AT(STRING_VALUE, '\"') and STOPS_AT(INT_VALUE, ",") and INT(INT_VALUE)
    '''


@lmql.query
async def create_chat_completion_gpt3(messages):
    '''
    argmax
        for message in messages:
            if message['role'] == 'system':
                "{:system} {message['content']}"
            elif message['role'] == 'user':
                "{:user} {message['content']}"
            elif message['role'] == 'assistant':
                "{:assistant} {message['content']}"
            else:
                assert False, "not a supported role " + str(role)
        schema = {
            "thoughts": {
                "text": str,
                "reasoning": str,
                "plan": str,
                "criticism": str,
                "speak": str
            },
            "command": {
                "name": str,
                "args": {
                    "[STRING_VALUE]" : str
                }
            }
        }
        stack = [("", schema)]
        indent = ""
        dq = '"'
        while len(stack) > 0:
            t = stack.pop(0)
            if type(t) is tuple:
                k, key_type = t
                if k != "":
                    "{indent}{dq}{k}{dq}: "
                if key_type is str:
                     if stack[0] == "DEDENT":
                        '"[STRING_VALUE]\n'
                     else:
                        '"[STRING_VALUE],\n'
                elif key_type is int:
                     if stack[0] == "DEDENT":
                        "[INT_VALUE]\n"
                     else:
                        "[INT_VALUE],\n"
                elif type(key_type) is dict:
                    "{{\n"
                    indent += "    "
                    if len(stack) == 0 or stack[0] == "DEDENT":
                        stack = [(k, key_type[k]) for k in key_type.keys()] + ["DEDENT", "}\n"] + stack
                    else:
                        stack = [(k, key_type[k]) for k in key_type.keys()] + ["DEDENT", "},\n"] + stack
                else:
                    assert False, "not a supported type " + str(k)
            elif t == "DEDENT":
                indent = indent[:-4]
            elif type(t) is str:
                "{indent}{t}"
            else:
                assert False, "not a supported type" + str(t)
    from
       'openai/gpt-3.5-turbo'
    where
       STOPS_AT(STRING_VALUE, '\"') and STOPS_AT(INT_VALUE, ",") and INT(INT_VALUE)
    '''
