"use client";

import { GraphTriggerInfo } from "@/app/api/__generated__/models/graphTriggerInfo";
import { LibraryAgentPreset } from "@/app/api/__generated__/models/libraryAgentPreset";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { CopyIcon } from "@phosphor-icons/react";
import { RunDetailCard } from "../../RunDetailCard/RunDetailCard";

interface Props {
  template: LibraryAgentPreset;
  triggerSetupInfo: GraphTriggerInfo;
}

function getTriggerStatus(
  template: LibraryAgentPreset,
): "active" | "inactive" | "broken" {
  if (!template.webhook_id || !template.webhook) return "broken";
  return template.is_active ? "active" : "inactive";
}

export function WebhookTriggerCard({ template, triggerSetupInfo }: Props) {
  const status = getTriggerStatus(template);
  const webhook = template.webhook;

  function handleCopyWebhookUrl() {
    if (webhook?.url) {
      navigator.clipboard.writeText(webhook.url);
    }
  }

  return (
    <RunDetailCard title="Trigger Status">
      <div className="flex flex-col gap-4">
        <div className="flex items-center gap-2">
          <Text variant="large-medium">Status</Text>
          <span
            className={`rounded-full px-2 py-0.5 text-xs font-medium ${
              status === "active"
                ? "bg-green-100 text-green-800"
                : status === "inactive"
                  ? "bg-yellow-100 text-yellow-800"
                  : "bg-red-100 text-red-800"
            }`}
          >
            {status === "active"
              ? "Active"
              : status === "inactive"
                ? "Inactive"
                : "Broken"}
          </span>
        </div>

        {!template.webhook_id ? (
          <Text variant="body" className="text-red-600">
            This trigger is not attached to a webhook. Use &quot;Set up
            trigger&quot; to fix this.
          </Text>
        ) : !triggerSetupInfo.credentials_input_name && webhook ? (
          <div className="flex flex-col gap-2">
            <Text variant="body">
              This trigger is ready to be used. Use the Webhook URL below to set
              up the trigger connection with the service of your choosing.
            </Text>
            <div className="flex flex-col gap-1">
              <Text variant="body-medium">Webhook URL:</Text>
              <div className="flex gap-2 rounded-md bg-gray-50 p-2">
                <code className="flex-1 select-all text-sm">{webhook.url}</code>
                <Button
                  variant="outline"
                  size="icon"
                  className="size-7 flex-none p-1"
                  onClick={handleCopyWebhookUrl}
                  title="Copy webhook URL"
                >
                  <CopyIcon className="size-4" />
                </Button>
              </div>
            </div>
          </div>
        ) : (
          <Text variant="body" className="text-muted-foreground">
            This agent trigger is{" "}
            {template.is_active
              ? "ready. When a trigger is received, it will run with the provided settings."
              : "disabled. It will not respond to triggers until you enable it."}
          </Text>
        )}
      </div>
    </RunDetailCard>
  );
}
