import React from "react";
import { Text } from "@/components/atoms/Text/Text";

interface ErrorMessageProps {
  isHttpError: boolean;
  errorMessage: string;
  context: string;
}

export function ErrorMessage({
  isHttpError,
  errorMessage,
  context,
}: ErrorMessageProps) {
  return (
    <div className="space-y-2">
      {isHttpError ? (
        <Text variant="body" className="text-zinc-700">
          {errorMessage}
        </Text>
      ) : (
        <>
          <Text variant="body" className="text-zinc-700">
            We had the following error when retrieving {context ?? "your data"}:
          </Text>
          <div className="rounded-lg border border-zinc-100 bg-zinc-50 p-3">
            <Text variant="body" className="!text-red-700">
              {errorMessage}
            </Text>
          </div>
        </>
      )}
    </div>
  );
}
