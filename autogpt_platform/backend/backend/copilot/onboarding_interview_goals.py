"""What Otto's onboarding interview learns, what it is not for, and what its
end signal carries.

The typed interview's prompt (``onboarding_interview``) takes what it learns
and its scope guard from here, and its end signal takes
:data:`END_SIGNAL_PARAMETERS`.

A leaf module with no imports.
"""

# The five things an interview learns, and how a task-first opener maps onto
# them.
_GOALS = """\
Learn these, skipping anything they've already told you:
1. What they do: their role and what the business or project is.
2. Who they work with or for: team size and customers.
3. Where their week goes: the recurring work that eats their time.
4. What they want handed off first.
5. The tools that work runs on, such as email, calendar, CRM, spreadsheets or social accounts.

If they open with a task ("make me a video for…", "build me an agent that…"), that task is the first thing to hand off. Say you've got it, then ask only what you need to staff it: what the business is, who it's for, and what they use today."""

# The scope guard: an interview takes in the person's work and does nothing
# else, whatever they ask for. Two redirects, then the interview ends.
_SCOPE_GUARD = """\
This conversation is only for learning about their work. Don't do tasks during it: no writing, coding, research, advice or answers to general questions. When they ask for something, treat it as a task for their team: say so in one short sentence, without explaining why, and go straight to your next question about their work. If they push past two redirects without telling you about their work, sum up whatever you have and say goodbye.

If they ask for something harmful, such as getting into someone else's account or tracking down a private person, say in one sentence that you can't help with that, then steer back to their own work."""

ONBOARDING_INTERVIEW_GOALS = f"{_GOALS}\n\n{_SCOPE_GUARD}"

# What an end signal carries: the task the person agreed their team starts on.
# The hand-off reads only the person's own words, and a bare "yes, that's
# right" does not say what was agreed, so Otto names it here.
END_SIGNAL_FIRST_TASK = "first_task"

# The input the end signal (``end_interview``) takes, as a JSON schema.
END_SIGNAL_PARAMETERS: dict[str, object] = {
    "type": "object",
    "properties": {
        END_SIGNAL_FIRST_TASK: {
            "type": "string",
            "description": (
                "The first thing their team will start on, as they agreed it "
                "(with any correction they made), in a few words, in their "
                "language."
            ),
        },
    },
    "required": [END_SIGNAL_FIRST_TASK],
    "additionalProperties": False,
}
