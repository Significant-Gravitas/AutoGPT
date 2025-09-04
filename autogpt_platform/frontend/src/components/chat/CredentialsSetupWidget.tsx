"use client";

import React, { useState } from "react";
import { Button } from "@/components/atoms/Button/Button";
import { Key, CheckCircle, XCircle, ExternalLink } from "lucide-react";
import { cn } from "@/lib/utils";

interface CredentialsSetupWidgetProps {
  _agentId: string;
  configuredCredentials: string[];
  missingCredentials: string[];
  totalRequired: number;
  message?: string;
  onSetupCredential?: (provider: string) => void;
  className?: string;
}

const PROVIDER_INFO: Record<
  string,
  { name: string; icon?: string; color: string }
> = {
  github: { name: "GitHub", color: "bg-gray-800" },
  google: { name: "Google", color: "bg-blue-500" },
  slack: { name: "Slack", color: "bg-purple-600" },
  notion: { name: "Notion", color: "bg-black" },
  discord: { name: "Discord", color: "bg-indigo-600" },
  openai: { name: "OpenAI", color: "bg-green-600" },
  anthropic: { name: "Anthropic", color: "bg-orange-600" },
  twitter: { name: "Twitter", color: "bg-sky-500" },
  linkedin: { name: "LinkedIn", color: "bg-blue-700" },
  default: { name: "API Key", color: "bg-neutral-600" },
};

export function CredentialsSetupWidget({
  _agentId,
  configuredCredentials,
  missingCredentials,
  totalRequired,
  message,
  onSetupCredential,
  className,
}: CredentialsSetupWidgetProps) {
  const [settingUp, setSettingUp] = useState<string | null>(null);

  const handleSetupCredential = (provider: string) => {
    setSettingUp(provider);
    if (onSetupCredential) {
      onSetupCredential(provider);
    }
    // In real implementation, this would open a modal or redirect to credentials page
    setTimeout(() => setSettingUp(null), 2000); // Simulate setup
  };

  const getProviderInfo = (provider: string) => {
    return PROVIDER_INFO[provider.toLowerCase()] || PROVIDER_INFO.default;
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
            <Key className="h-5 w-5 text-white" />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">
              Credentials Required
            </h3>
            <p className="text-sm text-neutral-600 dark:text-neutral-400">
              {message ||
                `Configure ${missingCredentials.length} credential${missingCredentials.length !== 1 ? "s" : ""} to use this agent`}
            </p>
          </div>
        </div>

        {/* Progress indicator */}
        <div className="mb-4">
          <div className="mb-2 flex items-center justify-between text-xs text-neutral-600 dark:text-neutral-400">
            <span>Setup Progress</span>
            <span>
              {configuredCredentials.length} of {totalRequired} configured
            </span>
          </div>
          <div className="h-2 overflow-hidden rounded-full bg-neutral-200 dark:bg-neutral-700">
            <div
              className="h-full bg-gradient-to-r from-green-500 to-emerald-500 transition-all duration-500"
              style={{
                width: `${(configuredCredentials.length / totalRequired) * 100}%`,
              }}
            />
          </div>
        </div>

        {/* Credentials list */}
        <div className="space-y-3">
          {/* Configured credentials */}
          {configuredCredentials.length > 0 && (
            <div>
              <p className="mb-2 text-xs font-medium text-green-700 dark:text-green-400">
                Configured
              </p>
              {configuredCredentials.map((credential) => {
                const info = getProviderInfo(credential);
                return (
                  <div
                    key={credential}
                    className="mb-2 flex items-center justify-between rounded-md bg-green-50 p-3 dark:bg-green-950/30"
                  >
                    <div className="flex items-center gap-3">
                      <div
                        className={cn(
                          "flex h-8 w-8 items-center justify-center rounded-md text-white",
                          info.color,
                        )}
                      >
                        <span className="text-xs font-bold">
                          {info.name.charAt(0)}
                        </span>
                      </div>
                      <span className="text-sm font-medium text-neutral-900 dark:text-neutral-100">
                        {info.name}
                      </span>
                    </div>
                    <CheckCircle className="h-5 w-5 text-green-600" />
                  </div>
                );
              })}
            </div>
          )}

          {/* Missing credentials */}
          {missingCredentials.length > 0 && (
            <div>
              <p className="mb-2 text-xs font-medium text-amber-700 dark:text-amber-400">
                Need Setup
              </p>
              {missingCredentials.map((credential) => {
                const info = getProviderInfo(credential);
                const isSettingUp = settingUp === credential;

                return (
                  <div
                    key={credential}
                    className="mb-2 flex items-center justify-between rounded-md bg-white/50 p-3 dark:bg-neutral-900/50"
                  >
                    <div className="flex items-center gap-3">
                      <div
                        className={cn(
                          "flex h-8 w-8 items-center justify-center rounded-md text-white",
                          info.color,
                        )}
                      >
                        <span className="text-xs font-bold">
                          {info.name.charAt(0)}
                        </span>
                      </div>
                      <div>
                        <span className="text-sm font-medium text-neutral-900 dark:text-neutral-100">
                          {info.name}
                        </span>
                        <p className="text-xs text-neutral-500">
                          {credential.includes("oauth")
                            ? "OAuth Connection"
                            : "API Key Required"}
                        </p>
                      </div>
                    </div>
                    <Button
                      onClick={() => handleSetupCredential(credential)}
                      variant="secondary"
                      size="sm"
                      disabled={isSettingUp}
                      className="min-w-[80px]"
                    >
                      {isSettingUp ? (
                        <span className="flex items-center gap-1">
                          <span className="h-3 w-3 animate-spin rounded-full border-2 border-current border-t-transparent" />
                          Setting up...
                        </span>
                      ) : (
                        <>
                          Connect
                          <ExternalLink className="ml-1 h-3 w-3" />
                        </>
                      )}
                    </Button>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        <div className="mt-4 flex items-center gap-2 rounded-md bg-amber-100 p-3 text-xs text-amber-700 dark:bg-amber-900/30 dark:text-amber-300">
          <XCircle className="h-4 w-4 flex-shrink-0" />
          <span>
            You need to configure all required credentials before this agent can
            be set up.
          </span>
        </div>
      </div>
    </div>
  );
}
