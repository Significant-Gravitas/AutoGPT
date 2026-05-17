"use client";

import type { SetupRequirementsResponse } from "@/app/api/__generated__/models/setupRequirementsResponse";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { CredentialsGroupedView } from "@/components/contextual/CredentialsInput/components/CredentialsGroupedView/CredentialsGroupedView";
import { FormRenderer } from "@/components/renderers/InputRenderer/FormRenderer";
import type { CredentialsMetaInput } from "@/lib/autogpt-server-api/types";
import { useEffect, useMemo, useState } from "react";
import {
  useAreAllConnected,
  useConnectedProvidersStore,
} from "../../connectedProvidersStore";
import { useCopilotChatActions } from "../CopilotChatActionsProvider/useCopilotChatActions";
import {
  ContentBadge,
  ContentCardDescription,
  ContentCardTitle,
  ContentMessage,
} from "../ToolAccordion/AccordionContent";
import {
  buildExpectedInputsSchema,
  buildPreviewRunMessage,
  buildRunMessage,
  buildSiblingInputsFromCredentials,
  checkAllCredentialsComplete,
  checkAllInputsComplete,
  checkCanRun,
  coerceCredentialFields,
  coerceExpectedInputs,
  extractInitialValues,
  getRequestedProviders,
  mergeInputValues,
} from "./helpers";

/**
 * Single credential/setup card rendered inline in copilot chats.
 *
 * Used by the run_block, run_agent and connect_integration tool renderers.
 * MCP has its own card (different OAuth route) and lives separately.
 *
 * - `inputsMode = "edit"` (default): renders inputs as an editable RJSF form,
 *   and `Proceed` sends the form values back to the chat. Used by run_block
 *   and connect_integration.
 * - `inputsMode = "preview"`: renders inputs as a read-only list
 *   (name • type, Required/Optional badge). Used by run_agent because graph
 *   inputs are set in the graph definition, not from the chat.
 */
interface Props {
  output: SetupRequirementsResponse;
  retryInstruction?: string;
  credentialsLabel?: string;
  inputsMode?: "edit" | "preview";
  onComplete?: () => void;
}

export function SetupRequirementsCard({
  output,
  retryInstruction,
  credentialsLabel,
  inputsMode = "edit",
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

  const isEditMode = inputsMode === "edit";

  const hasAdvancedFields =
    isEditMode && expectedInputs.some((i) => i.advanced);
  const inputSchema = isEditMode
    ? buildExpectedInputsSchema(expectedInputs, showAdvanced)
    : null;

  // Build siblingInputs for credential modal host prefill. In edit mode we also
  // include the live form values; in preview mode there is no form so only
  // discriminator-derived values contribute.
  const siblingInputs = useMemo(() => {
    const fromCreds = buildSiblingInputsFromCredentials(
      output.setup_info.user_readiness?.missing_credentials,
    );
    return isEditMode ? { ...inputValues, ...fromCreds } : fromCreds;
  }, [
    output.setup_info.user_readiness?.missing_credentials,
    inputValues,
    isEditMode,
  ]);

  function handleCredentialChange(key: string, value?: CredentialsMetaInput) {
    setInputCredentials((prev) => ({ ...prev, [key]: value }));
  }

  const needsCredentials = credentialFields.length > 0;
  const isAllCredsComplete = checkAllCredentialsComplete(
    requiredCredentials,
    inputCredentials,
  );

  const needsInputs = expectedInputs.length > 0;
  const isAllInputsDone = isEditMode
    ? checkAllInputsComplete(expectedInputs, inputValues)
    : true;

  // Session-scoped dismissal: when an earlier card in the same chat already
  // satisfied every provider this card asks for AND the card has nothing
  // else for the user to do, treat it as already sent. The `hasUserActionableInputs`
  // gate keeps the card visible when there are RJSF inputs the user still
  // needs to fill in (edit mode) — otherwise the chat hangs because
  // `handleRun` never fires and no message gets sent.
  const sessionID = output.session_id ?? null;
  const requestedProviders = getRequestedProviders(credentialFields);
  const alreadyConnected = useAreAllConnected(sessionID, requestedProviders);
  const hasUserActionableInputs = isEditMode && needsInputs;
  const canAutoDismiss =
    needsCredentials && alreadyConnected && !hasUserActionableInputs;

  // Auto-send when dismissing so the AI receives the run message and the
  // chat doesn't hang waiting for a confirmation that the user can no longer
  // provide (the Proceed button is hidden behind the early return below).
  //
  // `tryClaimAutoDismiss` provides the atomicity guarantee: only the first
  // card per `(sessionID, providers)` slot wins the claim and runs
  // `handleRun`; later cards (whether parallel siblings or StrictMode's
  // dev-mode re-mount of the same card) see `claimed === false`, set their
  // local `hasSent`, and dismiss silently — no chat spam, no double-send.
  // We deliberately call `handleRun` synchronously: a microtask + cleanup
  // pattern would leak the claim across StrictMode's double-invoke
  // (cleanup cancels the first microtask, but the claim is still held, so
  // the second effect run can't re-claim and the send never fires).
  useEffect(() => {
    if (!canAutoDismiss || hasSent) return;
    if (!sessionID || requestedProviders.length === 0) return;
    const claimed = useConnectedProvidersStore
      .getState()
      .tryClaimAutoDismiss({ sessionID, providers: requestedProviders });
    if (!claimed) {
      setHasSent(true);
      return;
    }
    handleRun();
    // eslint-disable-next-line react-hooks/exhaustive-deps -- handleRun captures latest state; claim guards re-entry
  }, [canAutoDismiss, hasSent]);

  if (hasSent || canAutoDismiss) {
    return <ContentMessage>Connected. Continuing…</ContentMessage>;
  }

  const canRun = checkCanRun(
    needsCredentials,
    isAllCredsComplete,
    isAllInputsDone,
  );

  function handleRun() {
    setHasSent(true);
    if (sessionID && requestedProviders.length > 0) {
      useConnectedProvidersStore
        .getState()
        .markConnected({ sessionID, providers: requestedProviders });
    }
    onComplete?.();
    const message = isEditMode
      ? buildRunMessage(
          needsCredentials,
          needsInputs,
          inputValues,
          retryInstruction,
        )
      : buildPreviewRunMessage(needsCredentials);
    onSend(message);
    if (isEditMode) setInputValues({});
  }

  return (
    <div className="grid gap-2">
      <ContentMessage>{output.message}</ContentMessage>

      {needsCredentials && (
        <div className="rounded-2xl border bg-background p-3">
          <Text variant="small" className="w-fit border-b text-zinc-500">
            {credentialsLabel ??
              (isEditMode ? "Credentials" : "Agent credentials")}
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

      {isEditMode && (inputSchema || hasAdvancedFields) && (
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

      {!isEditMode && expectedInputs.length > 0 && (
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
                {input.description && (
                  <ContentCardDescription className="mt-1">
                    {input.description}
                  </ContentCardDescription>
                )}
              </div>
            ))}
          </div>
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
