import React from "react";
import { Text } from "@/components/atoms/Text/Text";
import { Key, ArrowRight } from "@phosphor-icons/react";
import { Button } from "@/components/atoms/Button/Button";
import { cn } from "@/lib/utils";

export interface CredentialsNeededPromptProps {
  provider: string;
  providerName: string;
  credentialType: string;
  title: string;
  message: string;
  onSetupCredentials: () => void;
  onCancel: () => void;
  className?: string;
}

export function CredentialsNeededPrompt({
  provider: _provider,
  providerName,
  credentialType,
  title,
  message,
  onSetupCredentials,
  onCancel,
  className,
}: CredentialsNeededPromptProps) {
  function getCredentialTypeLabel(type: string): string {
    switch (type) {
      case "api_key":
        return "API Key";
      case "oauth2":
        return "OAuth Connection";
      case "user_password":
        return "Username & Password";
      case "host_scoped":
        return "Custom Headers";
      default:
        return "Credentials";
    }
  }

  return (
    <div
      className={cn(
        "mx-4 my-2 flex flex-col gap-4 rounded-lg border border-orange-200 bg-orange-50 p-6 dark:border-orange-900 dark:bg-orange-950",
        className,
      )}
    >
      {/* Icon & Header */}
      <div className="flex items-start gap-4">
        <div className="flex h-12 w-12 flex-shrink-0 items-center justify-center rounded-full bg-orange-500">
          <Key size={24} weight="bold" className="text-white" />
        </div>
        <div className="flex-1">
          <Text
            variant="h3"
            className="mb-1 text-orange-900 dark:text-orange-100"
          >
            Credentials Required
          </Text>
          <Text variant="body" className="text-orange-700 dark:text-orange-300">
            {message}
          </Text>
        </div>
      </div>

      {/* Details */}
      <div className="rounded-md bg-orange-100 p-4 dark:bg-orange-900">
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <Text
              variant="small"
              className="font-semibold text-orange-900 dark:text-orange-100"
            >
              Provider:
            </Text>
            <Text
              variant="body"
              className="text-orange-800 dark:text-orange-200"
            >
              {providerName}
            </Text>
          </div>
          <div className="flex items-center justify-between">
            <Text
              variant="small"
              className="font-semibold text-orange-900 dark:text-orange-100"
            >
              Type:
            </Text>
            <Text
              variant="body"
              className="text-orange-800 dark:text-orange-200"
            >
              {getCredentialTypeLabel(credentialType)}
            </Text>
          </div>
          <div className="flex items-center justify-between">
            <Text
              variant="small"
              className="font-semibold text-orange-900 dark:text-orange-100"
            >
              Needed for:
            </Text>
            <Text
              variant="body"
              className="text-orange-800 dark:text-orange-200"
            >
              {title}
            </Text>
          </div>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="flex gap-3">
        <Button
          onClick={onSetupCredentials}
          variant="primary"
          className="flex flex-1 items-center justify-center gap-2"
        >
          Setup Credentials
          <ArrowRight size={20} weight="bold" />
        </Button>
        <Button onClick={onCancel} variant="secondary">
          Cancel
        </Button>
      </div>

      <Text
        variant="small"
        className="text-center text-orange-600 dark:text-orange-400"
      >
        You&apos;ll need to add {providerName} credentials to continue
      </Text>
    </div>
  );
}
