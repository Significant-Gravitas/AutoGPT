"use client";

import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import type {
  BlockIOSubSchema,
  CredentialsMetaInput,
} from "@/lib/autogpt-server-api/types";
import { CredentialsInput } from "../CredentialsInputs/CredentialsInputs";
import {
  getAgentCredentialsFields,
  getAgentInputFields,
  renderValue,
} from "./helpers";

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

  // Take actual input entries as leading; augment with schema from input fields.
  // TODO: ensure consistent ordering.
  const inputEntries =
    inputs &&
    Object.entries(inputs).map<[string, [BlockIOSubSchema | undefined, any]]>(
      ([k, v]) => [k, [inputFields[k], v]],
    );

  const hasInputs = inputEntries && inputEntries.length > 0;
  const hasCredentials = credentialInputs && credentialFieldEntries.length > 0;

  if (!hasInputs && !hasCredentials) {
    return <div className="text-neutral-600">No input for this run.</div>;
  }

  return (
    <div className="flex flex-col gap-6">
      {/* Regular inputs */}
      {hasInputs && (
        <div className="flex flex-col gap-4">
          {inputEntries.map(([key, [schema, value]]) => (
            <div key={key} className="flex flex-col gap-1.5">
              <label className="text-sm font-medium">
                {schema?.title || key}
              </label>
              <p className="whitespace-pre-wrap break-words text-sm text-neutral-700">
                {renderValue(value)}
              </p>
            </div>
          ))}
        </div>
      )}

      {/* Credentials */}
      {hasCredentials && (
        <div className="flex flex-col gap-6">
          {hasInputs && <div className="border-t border-neutral-200 pt-4" />}
          {credentialFieldEntries.map(([key, inputSubSchema]) => {
            const credential = credentialInputs![key];
            if (!credential) return null;

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
