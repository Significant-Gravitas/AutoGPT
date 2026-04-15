"use client";

import type { SetupRequirementsResponse } from "@/app/api/__generated__/models/setupRequirementsResponse";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { CredentialsGroupedView } from "@/components/contextual/CredentialsInput/components/CredentialsGroupedView/CredentialsGroupedView";
import { FormRenderer } from "@/components/renderers/InputRenderer/FormRenderer";
import type { CredentialsMetaInput } from "@/lib/autogpt-server-api/types";
import { useEffect, useMemo, useState } from "react";
import { useCopilotChatActions } from "../../../../components/CopilotChatActionsProvider/useCopilotChatActions";
import { ContentMessage } from "../../../../components/ToolAccordion/AccordionContent";
import {
  buildExpectedInputsSchema,
  buildRunMessage,
  buildSiblingInputsFromCredentials,
  checkAllCredentialsComplete,
  checkAllInputsComplete,
  checkCanRun,
  coerceCredentialFields,
  coerceExpectedInputs,
  extractInitialValues,
  mergeInputValues,
} from "./helpers";

interface Props {
  output: SetupRequirementsResponse;
  retryInstruction?: string;
  credentialsLabel?: string;
  onComplete?: () => void;
}

export function SetupRequirementsCard({
  output,
  retryInstruction,
  credentialsLabel,
  onComplete,
}: Props) {
  const { onSend } = useCopilotChatActions();

  const [inputCredentials, setInputCredentials] = useState<
    Record<string, CredentialsMetaInput | undefined>
  >({});
  const [hasSent, setHasSent] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);

  const { credentialFields, requiredCredentials } = coerceCredentialFields(
    output.setup_info.user_readiness?.missing_credentials,
  );

  const expectedInputs = coerceExpectedInputs(
    (output.setup_info.requirements as Record<string, unknown>)?.inputs,
  );

  const initialValues = useMemo(
    () => extractInitialValues(expectedInputs),
    // eslint-disable-next-line react-hooks/exhaustive-deps -- stabilise on the raw prop
    [output.setup_info.requirements],
  );

  const [inputValues, setInputValues] =
    useState<Record<string, unknown>>(initialValues);

  const initialValuesKey = JSON.stringify(initialValues);
  useEffect(() => {
    setInputValues((prev) => mergeInputValues(initialValues, prev));
    // eslint-disable-next-line react-hooks/exhaustive-deps -- sync when serialised values change
  }, [initialValuesKey]);

  const hasAdvancedFields = expectedInputs.some((i) => i.advanced);
  const inputSchema = buildExpectedInputsSchema(expectedInputs, showAdvanced);

  // Build siblingInputs for credential modal host prefill.
  // Prefer discriminator_values from the credential response, but also
  // include values from input_data (e.g. url field) so the host pattern
  // can be extracted even when discriminator_values is empty.
  const siblingInputs = useMemo(() => {
    const fromCreds = buildSiblingInputsFromCredentials(
      output.setup_info.user_readiness?.missing_credentials,
    );
    return { ...inputValues, ...fromCreds };
  }, [output.setup_info.user_readiness?.missing_credentials, inputValues]);

  function handleCredentialChange(key: string, value?: CredentialsMetaInput) {
    setInputCredentials((prev) => ({ ...prev, [key]: value }));
  }

  const needsCredentials = credentialFields.length > 0;
  const isAllCredsComplete = checkAllCredentialsComplete(
    requiredCredentials,
    inputCredentials,
  );

  const needsInputs = expectedInputs.length > 0;
  const isAllInputsDone = checkAllInputsComplete(expectedInputs, inputValues);

  if (hasSent) {
    return <ContentMessage>Connected. Continuing…</ContentMessage>;
  }

  const canRun = checkCanRun(
    needsCredentials,
    isAllCredsComplete,
    isAllInputsDone,
  );

  function handleRun() {
    setHasSent(true);
    onComplete?.();
    onSend(
      buildRunMessage(
        needsCredentials,
        needsInputs,
        inputValues,
        retryInstruction,
      ),
    );
    setInputValues({});
  }

  return (
    <div className="grid gap-2">
      <ContentMessage>{output.message}</ContentMessage>

      {needsCredentials && (
        <div className="rounded-2xl border bg-background p-3">
          <Text variant="small" className="w-fit border-b text-zinc-500">
            {credentialsLabel ?? "Credentials"}
          </Text>
          <div className="mt-6">
            <CredentialsGroupedView
              credentialFields={credentialFields}
              requiredCredentials={requiredCredentials}
              inputCredentials={inputCredentials}
              inputValues={siblingInputs}
              onCredentialChange={handleCredentialChange}
            />
          </div>
        </div>
      )}

      {(inputSchema || hasAdvancedFields) && (
        <div className="rounded-2xl border bg-background p-3 pt-4">
          <Text variant="small" className="w-fit border-b text-zinc-500">
            Inputs
          </Text>
          {inputSchema && (
            <FormRenderer
              jsonSchema={inputSchema}
              className="mb-3 mt-3"
              handleChange={(v) =>
                setInputValues((prev) => ({ ...prev, ...(v.formData ?? {}) }))
              }
              uiSchema={{
                "ui:submitButtonOptions": { norender: true },
              }}
              initialValues={inputValues}
              formContext={{
                showHandles: false,
                size: "small",
              }}
            />
          )}
          {hasAdvancedFields && (
            <button
              type="button"
              className="text-xs text-muted-foreground underline"
              onClick={() => setShowAdvanced((v) => !v)}
            >
              {showAdvanced ? "Hide advanced fields" : "Show advanced fields"}
            </button>
          )}
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
