from autogpt.core.utils.json_schema import JSONSchema

DEFAULT_RESPONSE_SCHEMA = JSONSchema(
    type=JSONSchema.Type.OBJECT,
    properties={
        "thoughts": JSONSchema(
            type=JSONSchema.Type.OBJECT,
            required=True,
            properties={
                "observations": JSONSchema(
                    description=(
                        "Relevant observations from your last action (if any)"
                    ),
                    type=JSONSchema.Type.STRING,
                    required=False,
                ),
                "text": JSONSchema(
                    description="Thoughts",
                    type=JSONSchema.Type.STRING,
                    required=True,
                ),
                "reasoning": JSONSchema(
                    type=JSONSchema.Type.STRING,
                    required=True,
                ),
                "self_criticism": JSONSchema(
                    description="Constructive self-criticism",
                    type=JSONSchema.Type.STRING,
                    required=True,
                ),
                "plan": JSONSchema(
                    description=(
                        "Short markdown-style bullet list that conveys the "
                        "long-term plan"
                    ),
                    type=JSONSchema.Type.STRING,
                    required=True,
                ),
                "speak": JSONSchema(
                    description="Summary of thoughts, to say to user",
                    type=JSONSchema.Type.STRING,
                    required=True,
                ),
            },
        ),
        "command": JSONSchema(
            type=JSONSchema.Type.OBJECT,
            required=True,
            properties={
                "name": JSONSchema(
                    type=JSONSchema.Type.STRING,
                    required=True,
                ),
                "args": JSONSchema(
                    type=JSONSchema.Type.OBJECT,
                    required=True,
                ),
            },
        ),
    },
)
