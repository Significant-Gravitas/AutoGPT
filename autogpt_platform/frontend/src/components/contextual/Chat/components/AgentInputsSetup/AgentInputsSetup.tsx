"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Card } from "@/components/atoms/Card/Card";
import { Text } from "@/components/atoms/Text/Text";
import { CredentialsInput } from "@/components/contextual/CredentialsInputs/CredentialsInputs";
import { RunAgentInputs } from "@/components/contextual/RunAgentInputs/RunAgentInputs";
import type {
  BlockIOCredentialsSubSchema,
  BlockIOSubSchema,
} from "@/lib/autogpt-server-api/types";
import { cn } from "@/lib/utils";
import { PlayIcon, WarningIcon } from "@phosphor-icons/react";
import { useMemo } from "react";
import { useAgentInputsSetup } from "./useAgentInputsSetup";

interface Props {
  agentName?: string;
  inputSchema: Record<string, BlockIOSubSchema>;
  credentialsSchema?: Record<string, BlockIOCredentialsSubSchema>;
  message: string;
  onRun: (
    inputs: Record<string, any>,
    credentials: Record<string, any>,
  ) => void;
  onCancel?: () => void;
  className?: string;
}

export function AgentInputsSetup({
  agentName,
  inputSchema,
  credentialsSchema,
  message,
  onRun,
  onCancel,
  className,
}: Props) {
  const { inputValues, setInputValue, credentialsValues, setCredentialsValue } =
    useAgentInputsSetup();

  const inputFields = Object.entries(inputSchema || {});
  const credentialFields = Object.entries(credentialsSchema || {});

  const allRequiredInputsAreSet = useMemo(() => {
    const requiredFields = Object.entries(inputSchema || {}).filter(
      ([_, schema]) => !schema.hidden,
    );
    return requiredFields.every(([key]) => {
      const value = inputValues[key];
      return value !== undefined && value !== null && value !== "";
    });
  }, [inputSchema, inputValues]);

  const allCredentialsAreSet = useMemo(() => {
    if (!credentialsSchema || Object.keys(credentialsSchema).length === 0) {
      return true;
    }
    return Object.keys(credentialsSchema).every(
      (key) => credentialsValues[key] !== undefined,
    );
  }, [credentialsSchema, credentialsValues]);

  const canRun = allRequiredInputsAreSet && allCredentialsAreSet;

  function handleRun() {
    if (canRun) {
      onRun(inputValues, credentialsValues);
    }
  }

  return (
    <Card
      className={cn(
        "mx-4 my-2 overflow-hidden border-blue-200 bg-blue-50",
        className,
      )}
    >
      <div className="flex items-start gap-4 p-6">
        <div className="flex h-12 w-12 flex-shrink-0 items-center justify-center rounded-full bg-blue-500">
          <WarningIcon size={24} weight="bold" className="text-white" />
        </div>
        <div className="flex-1">
          <Text variant="h3" className="mb-2 text-blue-900">
            {agentName ? `Configure ${agentName}` : "Agent Configuration"}
          </Text>
          <Text variant="body" className="mb-4 text-blue-700">
            {message}
          </Text>

          {inputFields.length > 0 && (
            <div className="mb-4 space-y-4">
              {inputFields.map(([key, schema]) => {
                if (schema.hidden) return null;
                const defaultValue = (schema as any).default;
                return (
                  <RunAgentInputs
                    key={key}
                    schema={schema}
                    value={inputValues[key] ?? defaultValue}
                    placeholder={schema.description}
                    onChange={(value) => setInputValue(key, value)}
                  />
                );
              })}
            </div>
          )}

          {credentialFields.length > 0 && (
            <div className="mb-4 space-y-4">
              {credentialFields.map(([key, schema]) => (
                <CredentialsInput
                  key={key}
                  schema={schema}
                  selectedCredentials={credentialsValues[key]}
                  onSelectCredentials={(value) =>
                    setCredentialsValue(key, value)
                  }
                  siblingInputs={inputValues}
                />
              ))}
            </div>
          )}

          <div className="flex gap-2">
            <Button
              variant="primary"
              size="small"
              onClick={handleRun}
              disabled={!canRun}
            >
              <PlayIcon className="mr-2 h-4 w-4" weight="bold" />
              Run Agent
            </Button>
            {onCancel && (
              <Button variant="outline" size="small" onClick={onCancel}>
                Cancel
              </Button>
            )}
          </div>
        </div>
      </div>
    </Card>
  );
}
