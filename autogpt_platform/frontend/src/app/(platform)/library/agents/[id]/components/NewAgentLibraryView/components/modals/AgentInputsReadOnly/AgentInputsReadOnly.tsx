"use client";

import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { Text } from "@/components/atoms/Text/Text";
import { CredentialsInput } from "@/components/contextual/CredentialsInput/CredentialsInput";
import { isSystemCredential } from "@/components/contextual/CredentialsInput/helpers";
import type { CredentialsMetaInput } from "@/lib/autogpt-server-api/types";
import { RunAgentInputs } from "../RunAgentInputs/RunAgentInputs";
import { getAgentCredentialsFields, getAgentInputFields } from "./helpers";

type Props = {
  agent: LibraryAgent;
  inputs?: Record<string, any> | null;
  credentialInputs?: Record<string, CredentialsMetaInput> | null;
};

export function AgentInputsReadOnly({
  agent,
  inputs,
  credentialInputs,
}: Props) {
  const inputFields = getAgentInputFields(agent);
  const credentialFieldEntries = Object.entries(
    getAgentCredentialsFields(agent),
  );

  const inputEntries =
    inputs &&
    Object.entries(inputs).map(([key, value]) => ({
      key,
      schema: inputFields[key],
      value,
    }));

  const hasInputs = inputEntries && inputEntries.length > 0;
  const hasCredentials = credentialInputs && credentialFieldEntries.length > 0;

  if (!hasInputs && !hasCredentials) {
    return (
      <Text variant="body" className="text-zinc-700">
        No input for this run.
      </Text>
    );
  }

  return (
    <div className="flex flex-col gap-6">
      {/* Regular inputs */}
      {hasInputs && (
        <div className="flex flex-col gap-4">
          {inputEntries.map(({ key, schema, value }) => {
            if (!schema) return null;

            return (
              <RunAgentInputs
                key={key}
                schema={schema}
                value={value}
                placeholder={schema.description}
                onChange={() => {}}
                readOnly={true}
              />
            );
          })}
        </div>
      )}

      {/* Credentials */}
      {hasCredentials && (
        <div className="flex flex-col gap-6">
          {hasInputs && <div className="border-t border-neutral-200 pt-4" />}
          {credentialFieldEntries.map(([key, inputSubSchema]) => {
            const credential = credentialInputs![key];
            if (!credential) return null;
            if (isSystemCredential(credential)) return null;

            return (
              <CredentialsInput
                key={key}
                schema={{ ...inputSubSchema, discriminator: undefined } as any}
                selectedCredentials={credential}
                onSelectCredentials={() => {}}
                readOnly={true}
              />
            );
          })}
        </div>
      )}
    </div>
  );
}
