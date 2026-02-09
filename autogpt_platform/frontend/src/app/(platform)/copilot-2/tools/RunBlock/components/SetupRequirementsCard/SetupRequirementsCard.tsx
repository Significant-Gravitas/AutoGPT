"use client";

import type { SetupRequirementsResponse } from "@/app/api/__generated__/models/setupRequirementsResponse";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { CredentialsGroupedView } from "@/components/contextual/CredentialsInput/components/CredentialsGroupedView/CredentialsGroupedView";
import { FormRenderer } from "@/components/renderers/InputRenderer/FormRenderer";
import type { CredentialsMetaInput } from "@/lib/autogpt-server-api/types";
import { AnimatePresence, motion } from "framer-motion";
import { useState } from "react";
import { useCopilotChatActions } from "../../../../components/CopilotChatActionsProvider/useCopilotChatActions";
import {
  ContentBadge,
  ContentCardDescription,
  ContentCardTitle,
  ContentMessage,
} from "../../../../components/ToolAccordion/AccordionContent";
import {
  buildExpectedInputsSchema,
  coerceCredentialFields,
  coerceExpectedInputs,
} from "./helpers";

interface Props {
  output: SetupRequirementsResponse;
}

export function SetupRequirementsCard({ output }: Props) {
  const { onSend } = useCopilotChatActions();

  const [inputCredentials, setInputCredentials] = useState<
    Record<string, CredentialsMetaInput | undefined>
  >({});
  const [hasSentCredentials, setHasSentCredentials] = useState(false);

  const [showInputForm, setShowInputForm] = useState(false);
  const [inputValues, setInputValues] = useState<Record<string, unknown>>({});

  const { credentialFields, requiredCredentials } = coerceCredentialFields(
    output.setup_info.user_readiness?.missing_credentials,
  );

  const expectedInputs = coerceExpectedInputs(
    (output.setup_info.requirements as Record<string, unknown>)?.inputs,
  );

  const inputSchema = buildExpectedInputsSchema(expectedInputs);

  function handleCredentialChange(key: string, value?: CredentialsMetaInput) {
    setInputCredentials((prev) => ({ ...prev, [key]: value }));
  }

  const isAllCredentialsComplete =
    credentialFields.length > 0 &&
    [...requiredCredentials].every((key) => !!inputCredentials[key]);

  function handleProceedCredentials() {
    setHasSentCredentials(true);
    onSend(
      "I've configured the required credentials. Please re-run the block now.",
    );
  }

  function handleRunWithInputs() {
    const nonEmpty = Object.fromEntries(
      Object.entries(inputValues).filter(
        ([, v]) => v !== undefined && v !== null && v !== "",
      ),
    );
    onSend(
      `Run the block with these inputs: ${JSON.stringify(nonEmpty, null, 2)}`,
    );
    setShowInputForm(false);
    setInputValues({});
  }

  return (
    <div className="grid gap-2">
      <ContentMessage>{output.message}</ContentMessage>

      {credentialFields.length > 0 && (
        <div className="rounded-2xl border bg-background p-3">
          <CredentialsGroupedView
            credentialFields={credentialFields}
            requiredCredentials={requiredCredentials}
            inputCredentials={inputCredentials}
            inputValues={{}}
            onCredentialChange={handleCredentialChange}
          />
          {isAllCredentialsComplete && !hasSentCredentials && (
            <Button
              variant="primary"
              size="small"
              className="mt-3 w-full"
              onClick={handleProceedCredentials}
            >
              Proceed
            </Button>
          )}
        </div>
      )}

      {inputSchema && (
        <div className="flex gap-2 pt-2">
          <Button
            variant="outline"
            size="small"
            className="w-fit"
            onClick={() => setShowInputForm((prev) => !prev)}
          >
            {showInputForm ? "Hide inputs" : "Fill in inputs"}
          </Button>
        </div>
      )}

      <AnimatePresence initial={false}>
        {showInputForm && inputSchema && (
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
            <div className="rounded-2xl border bg-background p-3 pt-4">
              <Text variant="body-medium">Block inputs</Text>
              <FormRenderer
                jsonSchema={inputSchema}
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

      {expectedInputs.length > 0 && !inputSchema && (
        <div className="rounded-2xl border bg-background p-3">
          <ContentCardTitle className="text-xs">
            Expected inputs
          </ContentCardTitle>
          <div className="mt-2 grid gap-2">
            {expectedInputs.map((input) => (
              <div key={input.name} className="rounded-xl border p-2">
                <div className="flex items-center justify-between gap-2">
                  <ContentCardTitle className="text-xs">
                    {input.title}
                  </ContentCardTitle>
                  <ContentBadge>
                    {input.required ? "Required" : "Optional"}
                  </ContentBadge>
                </div>
                <ContentCardDescription className="mt-1">
                  {input.name} &bull; {input.type}
                  {input.description ? ` \u2022 ${input.description}` : ""}
                </ContentCardDescription>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
