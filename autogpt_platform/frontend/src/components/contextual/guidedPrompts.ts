export const NEW_SKILL_PROMPT = `I want to teach you a new skill, with the goal being that you fully learn and understand my process, including its edge-cases and gotchas. When you have all the info needed from me (without making assumptions), I want you to run the tool to create a new skill with all this information.

To start, ask me what I want to teach you.`;

export const NEW_SCHEDULED_TASK_PROMPT = `I want to create a new scheduled task. You will need to collect the following information from me:
- The schedule frequency
- and all necessary context of the task itself

Don't make any assumptions and make sure that the task instructions are unambiguous. You should also ask me whether or not each scheduled task should start a new session or continue in the same session.

Start by asking me about the task itself.`;

export function isGuidedPrompt(text: string) {
  return text === NEW_SKILL_PROMPT || text === NEW_SCHEDULED_TASK_PROMPT;
}
