"use client";

import { Renderer, type ActionEvent } from "@openuidev/react-lang";
import { ErrorBoundary } from "@/components/molecules/ErrorBoundary/ErrorBoundary";
import { autoGPTLibrary } from "./library";

interface Props {
  source: string;
  isStreaming: boolean;
  onAction: (event: ActionEvent) => void;
  revision: number;
}

export function OpenUI({ source, isStreaming, onAction, revision }: Props) {
  return (
    <ErrorBoundary
      key={revision}
      context="openui-experiment"
      fallback={
        <p
          role="alert"
          className="rounded-xl border border-red-200 bg-red-50 p-5 text-sm text-red-700"
        >
          This response could not be displayed. Try another request or inspect
          its source.
        </p>
      }
    >
      <Renderer
        response={source}
        library={autoGPTLibrary}
        isStreaming={isStreaming}
        onAction={onAction}
      />
    </ErrorBoundary>
  );
}
