"use client";

import { GraphTriggerInfo } from "@/app/api/__generated__/models/graphTriggerInfo";
import { LibraryAgentPreset } from "@/app/api/__generated__/models/libraryAgentPreset";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { RunDetailCard } from "../../RunDetailCard/RunDetailCard";
import { Copy01Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

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
                ? "bg-success/30 text-success-foreground"
                : status === "inactive"
                  ? // oxlint-disable-next-line design-system/no-raw-colors -- yellow-800 #876A00 has no role; olive to orange-brown change deferred (see PR)
                    "bg-warning-border text-yellow-800"
                  : "bg-error-surface text-error-foreground"
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
          <Text variant="body" tone="danger">
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
              <div className="flex gap-2 rounded-md bg-surface-muted p-2">
                <code className="ph-no-capture flex-1 text-sm select-all">
                  {webhook.url}
                </code>
                <Button
                  variant="outline"
                  size="icon"
                  // oxlint-disable-next-line design-system/no-restyle -- a 28px round copy button; icon-xs would make it a rounded square
                  className="size-7 flex-none p-1"
                  onClick={handleCopyWebhookUrl}
                  title="Copy webhook URL"
                >
                  <Icon icon={Copy01Icon} className="size-4" />
                </Button>
              </div>
            </div>
          </div>
        ) : (
          <Text variant="body" tone="muted">
            This workflow trigger is{" "}
            {template.is_active
              ? "ready. When a trigger is received, it will run with the provided settings."
              : "disabled. It will not respond to triggers until you enable it."}
          </Text>
        )}
      </div>
    </RunDetailCard>
  );
}
