"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { WarningDiamondIcon } from "@phosphor-icons/react";

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
    <div className="space-y-3 rounded-lg border border-red-200 bg-red-50 p-4">
      <div className="flex items-start gap-2">
        <WarningDiamondIcon
          size={20}
          weight="regular"
          className="mt-0.5 shrink-0 text-red-500"
        />
        <div className="flex-1 space-y-2">
          <Text variant="body-medium" className="text-red-900">
            {message || fallbackMessage}
          </Text>
          {error && (
            <details className="text-xs text-red-700">
              <summary className="cursor-pointer font-medium">
                Technical details
              </summary>
              <pre className="mt-2 max-h-40 overflow-auto whitespace-pre-wrap break-words rounded bg-red-100 p-2">
                {error}
              </pre>
            </details>
          )}
          {details && (
            <pre className="max-h-40 overflow-auto whitespace-pre-wrap break-words rounded bg-red-100 p-2 text-xs text-red-700">
              {details}
            </pre>
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
