"use client";

import React, { useState } from "react";
import { Button } from "@/components/atoms/Button/Button";
import { Key, CheckCircle, AlertCircle, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";
import { CredentialsInput } from "@/app/(platform)/library/agents/[id]/components/AgentRunsView/components/CredentialsInputs/CredentialsInputs";
import {
  BlockIOCredentialsSubSchema,
  CredentialsMetaInput,
} from "@/lib/autogpt-server-api/types";

interface CredentialsSetupWidgetProps {
  agentInfo: {
    id: string;
    name: string;
    graph_id?: string;
  };
  credentialsSchema: any; // This will be the credentials_input_schema from the agent
  onCredentialsSubmit?: (
    credentials: Record<string, CredentialsMetaInput>,
  ) => void;
  onSkip?: () => void;
  className?: string;
}

export function CredentialsSetupWidget({
  agentInfo,
  credentialsSchema,
  onCredentialsSubmit,
  onSkip,
  className,
}: CredentialsSetupWidgetProps) {
  const [selectedCredentials, setSelectedCredentials] = useState<
    Record<string, CredentialsMetaInput>
  >({});
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Parse the credentials schema to extract individual credential requirements
  // Handle both nested (with properties) and flat schema structures
  const schemaProperties =
    credentialsSchema?.properties || credentialsSchema || {};
  const credentialKeys = Object.keys(schemaProperties);
  const allCredentialsSelected = credentialKeys.every(
    (key) => selectedCredentials[key],
  );

  const handleCredentialSelect = (
    key: string,
    credential?: CredentialsMetaInput,
  ) => {
    if (credential) {
      setSelectedCredentials((prev) => ({
        ...prev,
        [key]: credential,
      }));
      setError(null);
    }
  };

  const handleSubmit = async () => {
    if (!allCredentialsSelected) {
      setError("Please provide all required credentials");
      return;
    }

    setIsSubmitting(true);
    setError(null);

    try {
      if (onCredentialsSubmit) {
        await onCredentialsSubmit(selectedCredentials);
      }
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to set up credentials",
      );
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div
      className={cn(
        "my-4 overflow-hidden rounded-lg border border-amber-200 dark:border-amber-800",
        "bg-gradient-to-br from-amber-50 to-orange-50 dark:from-amber-950/30 dark:to-orange-950/30",
        "duration-500 animate-in fade-in-50 slide-in-from-bottom-2",
        className,
      )}
    >
      <div className="px-6 py-5">
        <div className="mb-4 flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-full bg-amber-600">
            <AlertCircle className="h-5 w-5 text-white" />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">
              Credentials Required
            </h3>
            <p className="text-sm text-neutral-600 dark:text-neutral-400">
              The agent &quot;{agentInfo.name}&quot; requires credentials to
              run. Please provide the following:
            </p>
          </div>
        </div>

        {/* Credentials inputs */}
        <div className="space-y-4">
          {credentialKeys.map((key) => {
            const schema = schemaProperties[key] as BlockIOCredentialsSubSchema;
            const isSelected = !!selectedCredentials[key];

            return (
              <div
                key={key}
                className={cn(
                  "relative rounded-lg bg-white/50 p-4 dark:bg-neutral-900/50",
                  isSelected && "bg-green-50 dark:bg-green-950/30",
                )}
              >
                <CredentialsInput
                  schema={schema}
                  selectedCredentials={selectedCredentials[key]}
                  onSelectCredentials={(cred) =>
                    handleCredentialSelect(key, cred)
                  }
                  hideIfSingleCredentialAvailable={false}
                />
                {isSelected && (
                  <div className="absolute right-4 top-4">
                    <CheckCircle className="h-5 w-5 text-green-600" />
                  </div>
                )}
              </div>
            );
          })}
        </div>

        {error && (
          <div className="mt-4 rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-700 dark:border-red-800 dark:bg-red-950/30 dark:text-red-300">
            {error}
          </div>
        )}

        <div className="mt-6 flex gap-2">
          <Button
            onClick={handleSubmit}
            disabled={!allCredentialsSelected || isSubmitting}
            className="flex-1"
          >
            {isSubmitting ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Setting up...
              </>
            ) : (
              "Continue with Setup"
            )}
          </Button>
          {onSkip && (
            <Button variant="outline" onClick={onSkip} disabled={isSubmitting}>
              Skip for now
            </Button>
          )}
        </div>

        {!allCredentialsSelected && (
          <div className="mt-4 flex items-center gap-2 rounded-md bg-amber-100 p-3 text-xs text-amber-700 dark:bg-amber-900/30 dark:text-amber-300">
            <Key className="h-4 w-4 flex-shrink-0" />
            <span>
              You need to configure all required credentials before this agent
              can be set up.
            </span>
          </div>
        )}
      </div>
    </div>
  );
}
