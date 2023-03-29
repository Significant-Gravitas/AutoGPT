import json
import commands as cmd
import memory as mem
import data
import chat

def print_assistant_thoughts(assistant_reply):
    try:
        # Parse and print Assistant response
        assistant_reply_json = json.loads(assistant_reply)

        assistant_thoughts = assistant_reply_json.get("thoughts")
        if assistant_thoughts:
            assistant_thoughts_text = assistant_thoughts.get("text")
            assistant_thoughts_reasoning = assistant_thoughts.get("reasoning")
            assistant_thoughts_plan = assistant_thoughts.get("plan")
            assistant_thoughts_criticism = assistant_thoughts.get("criticism")
        else:
            assistant_thoughts_text = None
            assistant_thoughts_reasoning = None
            assistant_thoughts_plan = None
            assistant_thoughts_criticism = None
        
        print(f"ASSISTANT THOUGHTS: {assistant_thoughts_text}", flush=True)
        print(f"REASONING: {assistant_thoughts_reasoning}", flush=True)
        if assistant_thoughts_plan:
            print("PLAN: ", flush=True)
            if assistant_thoughts_plan:
                # Split the input_string using the newline character and dash
                lines = assistant_thoughts_plan.split('\n- ')

                # Iterate through the lines and print each one with a bullet point
                for line in lines:
                    print(f"- {line.strip()}", flush=True)
        print(f"CRITICISM: " + assistant_thoughts_criticism, flush=True)

    except json.decoder.JSONDecodeError:
        print("Error: Invalid JSON", flush=True)
        print(assistant_reply, flush=True)
    # All other errors, return "Error: + error message"
    except Exception as e:
        print("Error: " + str(e), flush=True)
        print(assistant_reply, flush=True)

# Initialize variables
full_message_history = []
prompt = data.load_prompt()
token_limit = 6000  # The maximum number of tokens allowed in the API call
result = None
user_input = "NEXT COMMAND"

# Interaction Loop
while True:
    # Send message to AI, get response
    assistant_reply = chat.chat_with_ai(prompt, user_input, full_message_history, mem.permanent_memory, token_limit)

    # Print Assistant thoughts
    print_assistant_thoughts(assistant_reply)

    # Get command name and arguments
    command_name, arguments = cmd.get_command(assistant_reply)
    
    ### GET USER AUTHORIZATION TO EXECUTE COMMAND ###
    # Get key press: Prompt the user to press enter to continue or escape to exit
    user_input = ""
    print(f"""NEXT ACTION: COMMAND = {command_name}  ARGUMENTS = {arguments}""", flush=True)
    print("Enter 'y' to authorise command or 'n' to exit program...", flush=True)
    while True:
        console_input = input()
        if console_input == "y":
            user_input = "NEXT COMMAND"
            break
        elif console_input == "n":
            user_input = "EXIT"
            break
        else:
            continue

    if user_input != "NEXT COMMAND":
        print("Exiting...", flush=True)
        break

    print("-=-=-=-=-=-=-= COMMAND AUTHORISED BY USER -=-=-=-=-=-=-=", flush=True)

    # Exectute command
    result = cmd.execute_command(command_name, arguments)
    

    # Check if there's a result from the command append it to the message history
    if result != None:
        full_message_history.append(chat.create_chat_message("system", result))
        print("system: " + result, flush=True)
    else:
        full_message_history.append(chat.create_chat_message("system", "Unable to execute command"))
        print("system: Unable to execute command", flush=True)
