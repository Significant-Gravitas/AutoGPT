"use client";

import type { SetupRequirementsResponse } from "@/app/api/__generated__/models/setupRequirementsResponse";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { CredentialsGroupedView } from "@/components/contextual/CredentialsInput/components/CredentialsGroupedView/CredentialsGroupedView";
import { FormRenderer } from "@/components/renderers/InputRenderer/FormRenderer";
import type { CredentialsMetaInput } from "@/lib/autogpt-server-api/types";
import { useState } from "react";
import { useCopilotChatActions } from "../../../../components/CopilotChatActionsProvider/useCopilotChatActions";
import { ContentMessage } from "../../../../components/ToolAccordion/AccordionContent";
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
  const [inputValues, setInputValues] = useState<Record<string, unknown>>({});
  const [hasSent, setHasSent] = useState(false);

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

  const needsCredentials = credentialFields.length > 0;
  const isAllCredentialsComplete =
    needsCredentials &&
    [...requiredCredentials].every((key) => !!inputCredentials[key]);

  const needsInputs = inputSchema !== null;
  const requiredInputNames = expectedInputs
    .filter((i) => i.required)
    .map((i) => i.name);
  const isAllInputsComplete =
    needsInputs &&
    requiredInputNames.every((name) => {
      const v = inputValues[name];
      return v !== undefined && v !== null && v !== "";
    });

  const canRun =
    !hasSent &&
    (!needsCredentials || isAllCredentialsComplete) &&
    (!needsInputs || isAllInputsComplete);

  function handleRun() {
    setHasSent(true);

    const parts: string[] = [];
    if (needsCredentials) {
      parts.push("I've configured the required credentials.");
    }

    if (needsInputs) {
      const nonEmpty = Object.fromEntries(
        Object.entries(inputValues).filter(
          ([, v]) => v !== undefined && v !== null && v !== "",
        ),
      );
      parts.push(
        `Run the block with these inputs: ${JSON.stringify(nonEmpty, null, 2)}`,
      );
    } else {
      parts.push("Please re-run the block now.");
    }

    onSend(parts.join(" "));
    setInputValues({});
  }

  return (
    <div className="grid gap-2">
      <ContentMessage>{output.message}</ContentMessage>

      {needsCredentials && (
        <div className="rounded-2xl border bg-background p-3">
          <Text variant="small" className="w-fit border-b text-zinc-500">
            Block credentials
          </Text>
          <div className="mt-6">
            <CredentialsGroupedView
              credentialFields={credentialFields}
              requiredCredentials={requiredCredentials}
              inputCredentials={inputCredentials}
              inputValues={{}}
              onCredentialChange={handleCredentialChange}
            />
          </div>
        </div>
      )}

      {inputSchema && (
        <div className="rounded-2xl border bg-background p-3 pt-4">
          <Text variant="small" className="w-fit border-b text-zinc-500">
            Block inputs
          </Text>
          <FormRenderer
            jsonSchema={inputSchema}
            className="mb-3 mt-3"
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
        </div>
      )}

      {(needsCredentials || needsInputs) && (
        <Button
          variant="primary"
          size="small"
          className="mt-4 w-fit"
          disabled={!canRun}
          onClick={handleRun}
        >
          Proceed
        </Button>
      )}
    </div>
  );
}
