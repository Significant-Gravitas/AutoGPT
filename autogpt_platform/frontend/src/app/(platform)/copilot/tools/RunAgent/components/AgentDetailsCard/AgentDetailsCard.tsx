"use client";

import type { AgentDetailsResponse } from "@/app/api/__generated__/models/agentDetailsResponse";
import { Button } from "@/components/atoms/Button/Button";
import { FormRenderer } from "@/components/renderers/InputRenderer/FormRenderer";
import { useState } from "react";
import { useCopilotChatActions } from "../../../../components/CopilotChatActionsProvider/useCopilotChatActions";
import { ContentMessage } from "../../../../components/ToolAccordion/AccordionContent";
import { buildInputSchema, extractDefaults, isFormValid } from "./helpers";

interface Props {
  output: AgentDetailsResponse;
}

export function AgentDetailsCard({ output }: Props) {
  const { onSend } = useCopilotChatActions();
  const schema = buildInputSchema(output.agent.inputs);

  const defaults = schema ? extractDefaults(schema) : {};

  const [inputValues, setInputValues] =
    useState<Record<string, unknown>>(defaults);
  const [valid, setValid] = useState(() =>
    schema ? isFormValid(schema, defaults) : false,
  );

  function handleChange(v: { formData?: Record<string, unknown> }) {
    const data = v.formData ?? {};
    setInputValues(data);
    if (schema) {
      setValid(isFormValid(schema, data));
    }
  }

  function handleProceed() {
    const nonEmpty = Object.fromEntries(
      Object.entries(inputValues).filter(
        ([, v]) => v !== undefined && v !== null && v !== "",
      ),
    );
    onSend(
      `Run the agent "${output.agent.name}" with these inputs: ${JSON.stringify(nonEmpty, null, 2)}`,
    );
  }

  if (!schema) {
    return (
      <div className="grid gap-2">
        <ContentMessage>This agent has no configurable inputs.</ContentMessage>
        <div className="flex gap-2 pt-2">
          <Button
            size="small"
            className="w-fit"
            onClick={() =>
              onSend(
                `Run the agent "${output.agent.name}" with placeholder/example values so I can test it.`,
              )
            }
          >
            Proceed
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="grid gap-2">
      <ContentMessage>
        Review the inputs below and press Proceed to run.
      </ContentMessage>

      <div className="mt-2 rounded-2xl border bg-background p-3 pt-4">
        <FormRenderer
          jsonSchema={schema}
          handleChange={handleChange}
          uiSchema={{
            "ui:submitButtonOptions": { norender: true },
          }}
          initialValues={inputValues}
          formContext={{
            showHandles: false,
            size: "small",
          }}
        />
      </div>

      <div className="mt-4">
        <Button
          variant="primary"
          size="small"
          className="w-fit"
          disabled={!valid}
          onClick={handleProceed}
        >
          Proceed
        </Button>
      </div>
    </div>
  );
}
