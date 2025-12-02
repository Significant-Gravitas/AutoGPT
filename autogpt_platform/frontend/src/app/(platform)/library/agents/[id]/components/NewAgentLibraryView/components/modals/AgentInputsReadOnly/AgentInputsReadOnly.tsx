"use client";

import React from "react";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import type { CredentialsMetaInput } from "@/lib/autogpt-server-api/types";
import { toDisplayName } from "@/providers/agent-credentials/helper";
import {
  getAgentCredentialsFields,
  getAgentInputFields,
  getCredentialTypeDisplayName,
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
  const fields = getAgentInputFields(agent);
  const credentialFields = getAgentCredentialsFields(agent);
  const inputEntries = Object.entries(fields);
  const credentialEntries = Object.entries(credentialFields);

  const hasInputs = inputs && inputEntries.length > 0;
  const hasCredentials = credentialInputs && credentialEntries.length > 0;

  if (!hasInputs && !hasCredentials) {
    return <div className="text-neutral-600">No input for this run.</div>;
  }

  return (
    <div className="flex flex-col gap-6">
      {/* Regular inputs */}
      {hasInputs && (
        <div className="flex flex-col gap-4">
          {inputEntries.map(([key, sub]) => (
            <div key={key} className="flex flex-col gap-1.5">
              <label className="text-sm font-medium">{sub?.title || key}</label>
              <p className="whitespace-pre-wrap break-words text-sm text-neutral-700">
                {renderValue((inputs as Record<string, any>)[key])}
              </p>
            </div>
          ))}
        </div>
      )}

      {/* Credentials */}
      {hasCredentials && (
        <div className="flex flex-col gap-6">
          {hasInputs && <div className="border-t border-neutral-200 pt-4" />}
          {credentialEntries.map(([key, _sub]) => {
            const credential = credentialInputs![key];
            if (!credential) return null;

            return (
              <div key={key} className="flex flex-col gap-4">
                <h3 className="text-lg font-medium text-neutral-900">
                  {toDisplayName(credential.provider)} credentials
                </h3>
                <div className="flex flex-col gap-3">
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-neutral-600">Name</span>
                    <span className="text-neutral-600">
                      {getCredentialTypeDisplayName(credential.type)}
                    </span>
                  </div>
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-neutral-900">
                      {credential.title || "Untitled"}
                    </span>
                    <span className="font-mono text-neutral-400">
                      {"*".repeat(25)}
                    </span>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
