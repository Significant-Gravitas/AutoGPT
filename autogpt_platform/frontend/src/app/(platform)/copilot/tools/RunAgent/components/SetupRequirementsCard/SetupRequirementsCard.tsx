"use client";

import type { SetupRequirementsResponse } from "@/app/api/__generated__/models/setupRequirementsResponse";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { CredentialsGroupedView } from "@/components/contextual/CredentialsInput/components/CredentialsGroupedView/CredentialsGroupedView";
import type { CredentialsMetaInput } from "@/lib/autogpt-server-api/types";
import { useState } from "react";
import { useCopilotChatActions } from "../../../../components/CopilotChatActionsProvider/useCopilotChatActions";
import {
  ContentBadge,
  ContentCardDescription,
  ContentCardTitle,
  ContentMessage,
} from "../../../../components/ToolAccordion/AccordionContent";
import { coerceCredentialFields, coerceExpectedInputs } from "./helpers";

interface Props {
  output: SetupRequirementsResponse;
}

export function SetupRequirementsCard({ output }: Props) {
  const { onSend } = useCopilotChatActions();

  const [inputCredentials, setInputCredentials] = useState<
    Record<string, CredentialsMetaInput | undefined>
  >({});
  const [hasSent, setHasSent] = useState(false);

  const { credentialFields, requiredCredentials } = coerceCredentialFields(
    output.setup_info.user_readiness?.missing_credentials,
  );

  const expectedInputs = coerceExpectedInputs(
    (output.setup_info.requirements as Record<string, unknown>)?.inputs,
  );

  function handleCredentialChange(key: string, value?: CredentialsMetaInput) {
    setInputCredentials((prev) => ({ ...prev, [key]: value }));
  }

  const needsCredentials = credentialFields.length > 0;
  const isAllCredentialsComplete =
    needsCredentials &&
    [...requiredCredentials].every((key) => !!inputCredentials[key]);

  const canProceed =
    !hasSent && (!needsCredentials || isAllCredentialsComplete);

  function handleProceed() {
    setHasSent(true);
    const message = needsCredentials
      ? "I've configured the required credentials. Please check if everything is ready and proceed with running the agent."
      : "Please proceed with running the agent.";
    onSend(message);
  }

  return (
    <div className="grid gap-2">
      <ContentMessage>{output.message}</ContentMessage>

      {needsCredentials && (
        <div className="rounded-2xl border bg-background p-3">
          <Text variant="small" className="w-fit border-b text-zinc-500">
            Agent credentials
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

      {expectedInputs.length > 0 && (
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

      {(needsCredentials || expectedInputs.length > 0) && (
        <Button
          variant="primary"
          size="small"
          className="mt-4 w-fit"
          disabled={!canProceed}
          onClick={handleProceed}
        >
          Proceed
        </Button>
      )}
    </div>
  );
}
