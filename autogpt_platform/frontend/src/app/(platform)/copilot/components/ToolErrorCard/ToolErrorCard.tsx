"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { AlertDiamondIcon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

interface Props {
  message?: string;
  fallbackMessage: string;
  error?: string;
  details?: string;
  actions: Array<{
    label: string;
    onClick: () => void;
    variant?: "outline" | "ghost";
  }>;
}

export function ToolErrorCard({
  message,
  fallbackMessage,
  error,
  details,
  actions,
}: Props) {
  return (
    <div className="space-y-3 rounded-lg bg-error-surface p-4">
      <div className="flex items-start gap-2">
        <Icon
          icon={AlertDiamondIcon}
          size={20}
          className="mt-0.5 shrink-0 text-error"
        />
        <div className="flex-1 space-y-2">
          <Text
            // TODO(tokens): no exact role for text-red-900
            variant="body-medium"
            className="text-red-900"
          >
            {message || fallbackMessage}
          </Text>
          {(error || details) && (
            <details className="text-xs text-error-foreground">
              <summary className="cursor-pointer font-medium">
                Technical details
              </summary>
              {error && (
                <pre
                  // TODO(tokens): no exact role for bg-red-100
                  className="mt-2 max-h-40 overflow-auto whitespace-pre-wrap break-words rounded bg-red-100 p-2"
                >
                  {error}
                </pre>
              )}
              {details && (
                <pre
                  // TODO(tokens): no exact role for bg-red-100
                  className="mt-2 max-h-40 overflow-auto whitespace-pre-wrap break-words rounded bg-red-100 p-2"
                >
                  {details}
                </pre>
              )}
            </details>
          )}
        </div>
      </div>
      <div className="flex gap-2 pt-3">
        {actions.map((action, i) => (
          <Button
            key={i}
            variant={action.variant ?? "outline"}
            size="small"
            onClick={action.onClick}
          >
            {action.label}
          </Button>
        ))}
      </div>
    </div>
  );
}
