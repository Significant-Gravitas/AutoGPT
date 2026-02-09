"use client";

import { useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Button } from "@/components/atoms/Button/Button";
import { FormRenderer } from "@/components/renderers/InputRenderer/FormRenderer";
import type { AgentDetailsResponse } from "@/app/api/__generated__/models/agentDetailsResponse";
import { useCopilotChatActions } from "../../../../components/CopilotChatActionsProvider/useCopilotChatActions";
import { buildInputSchema } from "./helpers";

interface Props {
  output: AgentDetailsResponse;
}

export function AgentDetailsCard({ output }: Props) {
  const { onSend } = useCopilotChatActions();
  const [showInputForm, setShowInputForm] = useState(false);
  const [inputValues, setInputValues] = useState<Record<string, unknown>>({});

  function handleRunWithExamples() {
    onSend(
      `Run the agent "${output.agent.name}" with placeholder/example values so I can test it.`,
    );
  }

  function handleRunWithInputs() {
    const nonEmpty = Object.fromEntries(
      Object.entries(inputValues).filter(
        ([, v]) => v !== undefined && v !== null && v !== "",
      ),
    );
    onSend(
      `Run the agent "${output.agent.name}" with these inputs: ${JSON.stringify(nonEmpty, null, 2)}`,
    );
    setShowInputForm(false);
    setInputValues({});
  }

  return (
    <div className="grid gap-2">
      <p className="text-sm text-foreground">
        Run this agent with example values or your own inputs.
      </p>

      <div className="flex gap-2 pt-4">
        <Button
          variant="outline"
          size="small"
          className="w-fit"
          onClick={handleRunWithExamples}
        >
          Run with example values
        </Button>
        <Button
          variant="secondary"
          size="small"
          className="w-fit"
          onClick={() => setShowInputForm((prev) => !prev)}
        >
          Run with my inputs
        </Button>
      </div>

      <AnimatePresence initial={false}>
        {showInputForm && buildInputSchema(output.agent.inputs) && (
          <motion.div
            initial={{ height: 0, opacity: 0, filter: "blur(6px)" }}
            animate={{ height: "auto", opacity: 1, filter: "blur(0px)" }}
            exit={{ height: 0, opacity: 0, filter: "blur(6px)" }}
            transition={{
              height: { type: "spring", bounce: 0.15, duration: 0.5 },
              opacity: { duration: 0.25 },
              filter: { duration: 0.2 },
            }}
            className="overflow-hidden"
            style={{ willChange: "height, opacity, filter" }}
          >
            <div className="mt-4 rounded-2xl border bg-background p-3 pt-4">
              <p className="text-sm font-medium text-foreground">
                Enter your inputs
              </p>
              <FormRenderer
                jsonSchema={buildInputSchema(output.agent.inputs)!}
                handleChange={(v) => setInputValues(v.formData ?? {})}
                uiSchema={{
                  "ui:submitButtonOptions": { norender: true },
                }}
                initialValues={inputValues}
                formContext={{
                  showHandles: false,
                  size: "small",
                }}
              />
              <div className="-mt-8 flex gap-2">
                <Button
                  variant="primary"
                  size="small"
                  className="w-fit"
                  onClick={handleRunWithInputs}
                >
                  Run
                </Button>
                <Button
                  variant="secondary"
                  size="small"
                  className="w-fit"
                  onClick={() => {
                    setShowInputForm(false);
                    setInputValues({});
                  }}
                >
                  Cancel
                </Button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
