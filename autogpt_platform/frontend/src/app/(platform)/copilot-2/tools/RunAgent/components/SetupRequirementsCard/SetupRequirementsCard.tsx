"use client";

import { useState } from "react";
import { CredentialsGroupedView } from "@/components/contextual/CredentialsInput/components/CredentialsGroupedView/CredentialsGroupedView";
import { Button } from "@/components/atoms/Button/Button";
import type { CredentialsMetaInput } from "@/lib/autogpt-server-api/types";
import type { SetupRequirementsResponse } from "@/app/api/__generated__/models/setupRequirementsResponse";
import { useCopilotChatActions } from "../../../../components/CopilotChatActionsProvider/useCopilotChatActions";
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

  const isAllComplete =
    credentialFields.length > 0 &&
    [...requiredCredentials].every((key) => !!inputCredentials[key]);

  function handleProceed() {
    setHasSent(true);
    onSend(
      "I've configured the required credentials. Please check if everything is ready and proceed with running the agent.",
    );
  }

  return (
    <div className="grid gap-2">
      <p className="text-sm text-foreground">{output.message}</p>

      {credentialFields.length > 0 && (
        <div className="rounded-2xl border bg-background p-3">
          <CredentialsGroupedView
            credentialFields={credentialFields}
            requiredCredentials={requiredCredentials}
            inputCredentials={inputCredentials}
            inputValues={{}}
            onCredentialChange={handleCredentialChange}
          />
          {isAllComplete && !hasSent && (
            <Button
              variant="primary"
              size="small"
              className="mt-3 w-full"
              onClick={handleProceed}
            >
              Proceed
            </Button>
          )}
        </div>
      )}

      {expectedInputs.length > 0 && (
        <div className="rounded-2xl border bg-background p-3">
          <p className="text-xs font-medium text-foreground">Expected inputs</p>
          <div className="mt-2 grid gap-2">
            {expectedInputs.map((input) => (
              <div key={input.name} className="rounded-xl border p-2">
                <div className="flex items-center justify-between gap-2">
                  <p className="truncate text-xs font-medium text-foreground">
                    {input.title}
                  </p>
                  <span className="shrink-0 rounded-full border bg-muted px-2 py-0.5 text-[11px] text-muted-foreground">
                    {input.required ? "Required" : "Optional"}
                  </span>
                </div>
                <p className="mt-1 text-xs text-muted-foreground">
                  {input.name} &bull; {input.type}
                  {input.description ? ` \u2022 ${input.description}` : ""}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
