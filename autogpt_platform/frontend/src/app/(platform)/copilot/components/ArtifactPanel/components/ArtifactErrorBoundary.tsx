"use client";

import * as Sentry from "@sentry/nextjs";
import { Component, type ErrorInfo, type ReactNode } from "react";
import { Text } from "@/components/atoms/Text/Text";

interface Props {
  children: ReactNode;
  artifactID: string;
  artifactTitle: string;
  artifactType: string;
}

interface State {
  error: Error | null;
}

export class ArtifactErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    Sentry.captureException(error, {
      contexts: {
        react: { componentStack: errorInfo.componentStack },
      },
      tags: { errorBoundary: "true", context: "copilot-artifact" },
      extra: {
        artifactID: this.props.artifactID,
        artifactTitle: this.props.artifactTitle,
        artifactType: this.props.artifactType,
      },
    });
  }

  componentDidUpdate(prevProps: Props) {
    if (
      this.state.error &&
      (prevProps.artifactID !== this.props.artifactID ||
        prevProps.artifactTitle !== this.props.artifactTitle ||
        prevProps.artifactType !== this.props.artifactType)
    ) {
      this.setState({ error: null });
    }
  }

  handleCopy = () => {
    const { error } = this.state;
    if (!error) return;
    const details = [
      `Artifact: ${this.props.artifactTitle}`,
      `ID: ${this.props.artifactID}`,
      `Type: ${this.props.artifactType}`,
      `Error: ${error.message}`,
      error.stack ? `Stack:\n${error.stack}` : "",
    ]
      .filter(Boolean)
      .join("\n");
    navigator.clipboard?.writeText(details).catch(() => {});
  };

  render() {
    const { error } = this.state;
    if (!error) return this.props.children;

    const message = error.message || "Unknown rendering error";

    return (
      <div
        role="alert"
        className="flex h-full flex-col items-center justify-center gap-3 p-8 text-center"
      >
        <Text variant="body-medium-tight" tone="body">
          This artifact couldn&apos;t be rendered
        </Text>
        <Text
          variant="small-tight"
          tone="muted"
          className="max-w-md break-words"
        >
          Something in{" "}
          <span className="font-mono">{this.props.artifactTitle}</span> threw an
          error while rendering. The chat and sidebar are still working.
        </Text>
        <pre className="max-h-32 max-w-md overflow-auto rounded-md bg-surface-hover px-3 py-2 text-left text-xs break-words whitespace-pre-wrap text-foreground-body">
          {message}
        </pre>
        <button
          type="button"
          onClick={this.handleCopy}
          className="rounded-md border border-border-base bg-surface px-3 py-1.5 text-xs font-medium text-foreground-body shadow-sm transition-colors hover:bg-surface-muted focus-visible:ring-2 focus-visible:ring-brand/60 focus-visible:outline-none"
        >
          Copy error details
        </button>
        <Text variant="small-tight" tone="subtle" className="max-w-md">
          Paste this into the chat so a working version can be regenerated.
        </Text>
      </div>
    );
  }
}
