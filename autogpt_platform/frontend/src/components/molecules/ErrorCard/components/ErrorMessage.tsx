import React from "react";
import { Text } from "@/components/atoms/Text/Text";

interface Props {
  errorMessage: string;
  context: string;
  hint?: string;
  textSize?: "body" | "body-md";
}

export function ErrorMessage({
  errorMessage,
  context,
  hint,
  textSize = "body",
}: Props) {
  return (
    <div data-slot="error-card-message" className="space-y-2">
      <Text
        variant={textSize}
        data-slot="error-card-context"
        className="text-foreground-body"
      >
        We had the following error when retrieving {context ?? "your data"}:
      </Text>
      <div
        data-slot="error-card-detail"
        className="rounded-lg border border-separator bg-surface-muted p-3"
      >
        <Text
          variant={textSize}
          data-slot="error-card-detail-text"
          className="!text-error-foreground"
        >
          {errorMessage}
        </Text>
      </div>
      {hint && (
        <div data-slot="error-card-hint" className="!mt-4">
          <Text
            variant={textSize}
            data-slot="error-card-hint-text"
            className="text-foreground-body"
          >
            {hint}
          </Text>
        </div>
      )}
    </div>
  );
}
