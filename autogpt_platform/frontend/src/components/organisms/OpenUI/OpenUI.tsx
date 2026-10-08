"use client";

import { Renderer, type ActionEvent } from "@openuidev/react-lang";
import { ErrorBoundary } from "@/components/molecules/ErrorBoundary/ErrorBoundary";
import { autoGPTLibrary } from "./library";
import { OpenUIInteractionContext } from "./interactionContext";
import type { ReactNode } from "react";

interface Props {
  source: string;
  isStreaming: boolean;
  onAction: (event: ActionEvent) => void;
  revision: number;
  disabled?: boolean;
  initialState?: Record<string, unknown>;
  onStateUpdate?: (state: Record<string, unknown>) => void;
  fallback?: ReactNode;
}

export function OpenUI({
  source,
  isStreaming,
  onAction,
  revision,
  disabled = false,
  initialState,
  onStateUpdate,
  fallback,
}: Props) {
  return (
    <OpenUIInteractionContext.Provider value={disabled || isStreaming}>
      <ErrorBoundary
        key={revision}
        context="copilot-openui"
        fallback={
          fallback ?? (
            <p
              role="alert"
              className="rounded-xl border border-red-200 bg-red-50 p-5 text-sm text-red-700"
            >
              This response could not be displayed. Try another request.
            </p>
          )
        }
      >
        <Renderer
          response={source}
          library={autoGPTLibrary}
          isStreaming={isStreaming}
          onAction={onAction}
          initialState={initialState}
          onStateUpdate={onStateUpdate}
        />
      </ErrorBoundary>
    </OpenUIInteractionContext.Provider>
  );
}
