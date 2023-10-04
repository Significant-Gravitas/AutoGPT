import runBot
import aiohttp
import os
import discord
import requests
import asyncio
import subprocess
import time
import json
import pty
from dotenv import load_dotenv
from discord import app_commands, SelectOption
from discord.ui import MentionableSelect
import logging

logging.basicConfig(level=logging.DEBUG)

load_dotenv()
API_KEY = os.getenv('API_KEY')
DISCORD_BOT_SECRET = os.getenv('DISCORD_BOT_TOKEN')
FLOWISE_API_URL = "https://flow.ikig-ai.me/api/v1/prediction/b72742a3-46d1-41bf-a493-46b86925e5be"
SURVEY_API_URL = "https://flow.ikig-ai.me/api/v1/prediction/1200e62b-68b8-4c18-86ba-a2f16e5cf085"
DAILY_TASKS_API_URL = "https://flow.ikig-ai.me/api/v1/prediction/8fdc6d93-9627-4f76-b008-96456ec591c2"
MONTHLY_TASKS_API_URL = "https://flow.ikig-ai.me/api/v1/prediction/0d341483-68d9-45e7-be50-0be0b5e2fe44"
GROUP_TASK_API_URL = "https://flow.ikig-ai.me/api/v1/prediction/57a1dba0-59fa-40e3-a749-330748b72012"
headers = {"Authorization": API_KEY}


class CustomClient(discord.Client):

    def __init__(self, *, intents: discord.Intents):
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)

    async def setup_hook(self):
        await self.tree.sync()


client = CustomClient(intents=discord.Intents.all())


async def query(api_url, payload):
    print(f"Debug: Making request to {api_url} with payload: {payload}")
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(api_url, headers=headers, data=payload) as response:
                result = await response.json()
                print(f"Debug: Response from {api_url}: {result}")
                return result
        except Exception as e:
            print(f"Debug: Error during request to {api_url}: {str(e)}")
            return {"error": str(e)}


async def survey(interaction):
    questions = [
        ("Section 1: Passions\nWhat activities make you lose track of time, and what hobbies or tasks do you love so much that you would do them for free?", "passions"),
        ("Section 2: Missions\nWhat causes or issues deeply resonate with you, and if you had all the resources in the world, what problem would you want to solve?", "missions"),
        ("Section 3: Professions\nWhat are your top three skills or talents, and if you could choose any job in the world, what would it be?", "professions"),
        ("Section 4: Vocations\nWhat skills or talents do you possess that people would be willing to pay for, and if you were to start a business or offer a service, what would it be?", "vocations"),
    ]

    input_array = []  # To hold the array of question-answer objects
    for question_text, section in questions:
        await interaction.followup.send(question_text)
        try:
            response = await client.wait_for('message', check=lambda message: message.author == interaction.user)
            # Appending the question and answer as an object
            input_array.append(
                {"question": question_text, "answer": response.content})
        except asyncio.TimeoutError:
            await interaction.followup.send('You took too long to answer. Please try the survey again.')
            return None
    # Convert the input_array to a JSON string and return
    return json.dumps(input_array)


async def configure_env(interaction):
    # Determine the channel type (guild channel or DM)
    channel_type = interaction.channel.type

    # Create a function to send messages (compatible with both DMs and guild channels)
    async def send_message(message):
        if channel_type == discord.ChannelType.private:
            await interaction.user.send(message)
        else:
            await interaction.followup.send(message)

    await interaction.response.defer()

    # Prompt the user for the OpenAI API key
    await send_message("Please provide your OpenAI API key:")

    def remove_last_line_from_file(filename):
        with open(filename, 'r+') as file:
            lines = file.readlines()
            file.truncate(0)  # Clear the file content
            file.seek(0)  # Move the pointer to the beginning of the file
            for line in lines[:-1]:  # Write all lines except the last one
                file.write(line)

    def check_key_message(message):
        return message.author == interaction.user and message.channel == interaction.channel

    # Wait for the user's response
    key_message = await client.wait_for('message', check=check_key_message)
    openai_key = key_message.content

    # Update the .env file in the specified directory
    env_file_path = '../autogpts/autogptold'

    # Using 'a' mode to append the key without overwriting existing content
    with open(env_file_path, 'a') as file:
        file.write(f"OPENAI_API_KEY={openai_key}\n")

    # Optionally, you can update the environment variables for the current process as well
    os.environ["OPENAI_API_KEY"] = openai_key

    await send_message("OpenAI API key has been set successfully!")


async def run_docker_commands(interaction):
    # Navigate to the specified directory
    os.chdir('../auto-ikigAI')

    # Build the Docker image
    build_process = subprocess.Popen(["docker-compose", "build", "auto-gpt"])
    build_process.wait()

    # Create a pseudo-terminal
    master, slave = pty.openpty()

    # Run the Docker container with the specified options
    run_process = subprocess.Popen(["docker-compose", "run", "-u", "root", "--rm", "auto-gpt", "--gpt4only", "--continuous"],
                                   stdin=slave, text=True)

    remove_last_line_from_file(env_file_path)

    # Wait for 1 minute
    await asyncio.sleep(30)

    # Write 'n' to the master side of the pseudo-terminal
    os.write(master, b'n\n')

# Prompt the user for their goal
    await interaction.followup.send('Please enter your goal:')
    try:
        response = await client.wait_for('message', timeout=120.0, check=lambda message: message.author == interaction.user)
        goal_prompt = response.content
        # Send the user's input to the Docker process
        os.write(master, (goal_prompt + '\n').encode())

        # Continue with the rest of the code as needed
    except asyncio.TimeoutError:
        await interaction.followup.send('You took too long to answer. Please try again.')


class GroupTaskSelect(MentionableSelect):
    async def callback(self, interaction: discord.Interaction):
        # Defer the interaction
        await interaction.response.defer()

        mentioned_users = self.values
        group_task_data = []
        for user in mentioned_users:
            discord_roles = [
                role.name for role in user.roles if role.name != "@everyone"]
            user_data = {
                "username": user.name,
                "roles": discord_roles
            }
            group_task_data.append(user_data)

        payload = json.dumps(group_task_data)

        print(f"Debug: Sending the following data to the API: {payload}")

        json.dumps(payload)

        async with aiohttp.ClientSession() as session:
            async with session.post(GROUP_TASK_API_URL, data=payload, headers=headers) as response:
                # or await response.json() if you expect JSON
                response_text = await response.text()

        # Use follow-up to send the response
        if response.status == 200:
            await interaction.followup.send("Group task created successfully!")
        else:
            await interaction.followup.send(f"Failed to create group task: {response.text}")


class GroupTaskView(discord.ui.View):
    def __init__(self):
        super().__init__()
        self.add_item(GroupTaskSelect(custom_id="users",
                      placeholder="Select users for group task", max_values=25))

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        return True  # Allowing all interactions, you can customize this logic if needed


@client.tree.command(name="configure_env", description="Configure environment variables")
async def invoke_configure_env_command(interaction: discord.Interaction):
    await configure_env(interaction)


@client.tree.command(name="grouptask", description="Assign group tasks to mentioned users")
async def group_task_command(interaction: discord.Interaction):
    try:
        await interaction.response.defer()
        view = GroupTaskView()
        await interaction.followup.send('Please select the users for the group task:', view=view)
    except Exception as e:
        print(f"An error occurred: {e}")
        await interaction.followup.send(f"An error occurred: {e}")


@client.event
async def on_ready():
    print("I'm in")
    print(client.user)


@client.tree.command(name="summon", description="Summon the AI to execute a specific task")
async def summon_command(interaction: discord.Interaction):
    await interaction.response.defer()
    await interaction.followup.send('Summoning the AI and executing the task, please wait...')
    await run_docker_commands(interaction)
    await interaction.followup.send('Task completed successfully!')


@client.tree.command(name="chat", description="Ask a question to Flowise")
async def chat(interaction: discord.Interaction, question: str):
    await interaction.response.defer()
    output = query(FLOWISE_API_URL, {"question": question})
    await interaction.followup.send(output)


@client.tree.command(name="survey", description="Take a survey")
async def survey_command(interaction: discord.Interaction):
    await interaction.response.defer()
    # Get the JSON string from the survey function
    answers_string = await survey(interaction)
    if answers_string:
        # Pass the JSON object to the query function
        survey_output = await query(SURVEY_API_URL, json.loads(answers_string))

        json.dumps(survey_output)  # Convert the output to a JSON string

        # Query the daily tasks API
        daily_tasks_output = await query(DAILY_TASKS_API_URL, survey_output)

        # Send the daily tasks result to the user
        daily_tasks_string = f"Daily Tasks: {json.dumps(daily_tasks_output)}"

        await interaction.followup.send(daily_tasks_string)

        await asyncio.sleep(2)

        # Query the monthly tasks API
        monthly_tasks_output = await query(MONTHLY_TASKS_API_URL, survey_output)

        # Send the monthly tasks result to the user
        monthly_tasks_string = f"Monthly Tasks: {json.dumps(monthly_tasks_output)}"

        await interaction.followup.send(monthly_tasks_string)

client.run(DISCORD_BOT_SECRET)

if __name__ == "__main__":
    runBot.runDiscordBot()



