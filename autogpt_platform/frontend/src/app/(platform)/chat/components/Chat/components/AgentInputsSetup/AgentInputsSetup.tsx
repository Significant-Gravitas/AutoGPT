"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Card } from "@/components/atoms/Card/Card";
import { Text } from "@/components/atoms/Text/Text";
import { CredentialsInput } from "@/components/contextual/CredentialsInput/CredentialsInput";
import { RunAgentInputs } from "@/components/contextual/RunAgentInputs/RunAgentInputs";

import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import {
  BlockIOCredentialsSubSchema,
  BlockIOSubSchema,
} from "@/lib/autogpt-server-api/types";
import { cn, isEmpty } from "@/lib/utils";
import { PlayIcon, WarningIcon } from "@phosphor-icons/react";
import { useMemo } from "react";
import { useAgentInputsSetup } from "./useAgentInputsSetup";

type LibraryAgentInputSchemaProperties = LibraryAgent["input_schema"] extends {
  properties: infer P;
}
  ? P extends Record<string, BlockIOSubSchema>
    ? P
    : Record<string, BlockIOSubSchema>
  : Record<string, BlockIOSubSchema>;

type LibraryAgentCredentialsInputSchemaProperties =
  LibraryAgent["credentials_input_schema"] extends {
    properties: infer P;
  }
    ? P extends Record<string, BlockIOCredentialsSubSchema>
      ? P
      : Record<string, BlockIOCredentialsSubSchema>
    : Record<string, BlockIOCredentialsSubSchema>;

interface Props {
  agentName?: string;
  inputSchema: LibraryAgentInputSchemaProperties | Record<string, any>;
  credentialsSchema?:
    | LibraryAgentCredentialsInputSchemaProperties
    | Record<string, any>;
  message: string;
  requiredFields?: string[];
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
  requiredFields,
  onRun,
  onCancel,
  className,
}: Props) {
  const { inputValues, setInputValue, credentialsValues, setCredentialsValue } =
    useAgentInputsSetup();

  const inputSchemaObj = useMemo(() => {
    if (!inputSchema) return { properties: {}, required: [] };
    if ("properties" in inputSchema && "type" in inputSchema) {
      return inputSchema as {
        properties: Record<string, any>;
        required?: string[];
      };
    }
    return { properties: inputSchema as Record<string, any>, required: [] };
  }, [inputSchema]);

  const credentialsSchemaObj = useMemo(() => {
    if (!credentialsSchema) return { properties: {}, required: [] };
    if ("properties" in credentialsSchema && "type" in credentialsSchema) {
      return credentialsSchema as {
        properties: Record<string, any>;
        required?: string[];
      };
    }
    return {
      properties: credentialsSchema as Record<string, any>,
      required: [],
    };
  }, [credentialsSchema]);

  const agentInputFields = useMemo(() => {
    const properties = inputSchemaObj.properties || {};
    return Object.fromEntries(
      Object.entries(properties).filter(
        ([_, subSchema]: [string, any]) => !subSchema.hidden,
      ),
    );
  }, [inputSchemaObj]);

  const agentCredentialsInputFields = useMemo(() => {
    return credentialsSchemaObj.properties || {};
  }, [credentialsSchemaObj]);

  const inputFields = Object.entries(agentInputFields);
  const credentialFields = Object.entries(agentCredentialsInputFields);

  const defaultsFromSchema = useMemo(() => {
    const defaults: Record<string, any> = {};
    Object.entries(agentInputFields).forEach(([key, schema]) => {
      if ("default" in schema && schema.default !== undefined) {
        defaults[key] = schema.default;
      }
    });
    return defaults;
  }, [agentInputFields]);

  const defaultsFromCredentialsSchema = useMemo(() => {
    const defaults: Record<string, any> = {};
    Object.entries(agentCredentialsInputFields).forEach(([key, schema]) => {
      if ("default" in schema && schema.default !== undefined) {
        defaults[key] = schema.default;
      }
    });
    return defaults;
  }, [agentCredentialsInputFields]);

  const mergedInputValues = useMemo(() => {
    return { ...defaultsFromSchema, ...inputValues };
  }, [defaultsFromSchema, inputValues]);

  const mergedCredentialsValues = useMemo(() => {
    return { ...defaultsFromCredentialsSchema, ...credentialsValues };
  }, [defaultsFromCredentialsSchema, credentialsValues]);

  const allRequiredInputsAreSet = useMemo(() => {
    const requiredInputs = new Set(
      requiredFields || (inputSchemaObj.required as string[]) || [],
    );
    const nonEmptyInputs = new Set(
      Object.keys(mergedInputValues).filter(
        (k) => !isEmpty(mergedInputValues[k]),
      ),
    );
    const missing = [...requiredInputs].filter(
      (input) => !nonEmptyInputs.has(input),
    );
    return missing.length === 0;
  }, [inputSchemaObj.required, mergedInputValues, requiredFields]);

  const allCredentialsAreSet = useMemo(() => {
    const requiredCredentials = new Set(
      (credentialsSchemaObj.required as string[]) || [],
    );
    if (requiredCredentials.size === 0) {
      return true;
    }
    const missing = [...requiredCredentials].filter((key) => {
      const cred = mergedCredentialsValues[key];
      return !cred || !cred.id;
    });
    return missing.length === 0;
  }, [credentialsSchemaObj.required, mergedCredentialsValues]);

  const canRun = allRequiredInputsAreSet && allCredentialsAreSet;

  function handleRun() {
    if (canRun) {
      onRun(mergedInputValues, mergedCredentialsValues);
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
              {inputFields.map(([key, inputSubSchema]) => (
                <RunAgentInputs
                  key={key}
                  schema={inputSubSchema}
                  value={inputValues[key] ?? inputSubSchema.default}
                  placeholder={inputSubSchema.description}
                  onChange={(value) => setInputValue(key, value)}
                />
              ))}
            </div>
          )}

          {credentialFields.length > 0 && (
            <div className="mb-4 space-y-4">
              {credentialFields.map(([key, schema]) => {
                const requiredCredentials = new Set(
                  (credentialsSchemaObj.required as string[]) || [],
                );
                return (
                  <CredentialsInput
                    key={key}
                    schema={schema}
                    selectedCredentials={credentialsValues[key]}
                    onSelectCredentials={(value) =>
                      setCredentialsValue(key, value)
                    }
                    siblingInputs={mergedInputValues}
                    isOptional={!requiredCredentials.has(key)}
                  />
                );
              })}
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
