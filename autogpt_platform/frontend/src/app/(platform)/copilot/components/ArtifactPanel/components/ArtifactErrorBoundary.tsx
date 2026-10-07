"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import * as Sentry from "@sentry/nextjs";
import { Component, type ErrorInfo, type ReactNode } from "react";

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
        <Text variant="body-medium" as="p" tone="secondary">
          This artifact couldn&apos;t be rendered
        </Text>
        <Text
          variant="small"
          as="p"
          tone="muted"
          unmask={false}
          className="max-w-md wrap-break-word"
        >
          Something in{" "}
          <span className="font-mono">{this.props.artifactTitle}</span> threw an
          error while rendering. The chat and sidebar are still working.
        </Text>
        <pre className="max-h-32 max-w-md overflow-auto rounded-md bg-zinc-100 px-3 py-2 text-left text-xs wrap-break-word whitespace-pre-wrap text-zinc-700">
          {message}
        </pre>
        <Button
          type="button"
          variant="secondary"
          size="sm"
          onClick={this.handleCopy}
          className="h-auto px-3 py-1.5 leading-4 text-zinc-700"
        >
          Copy error details
        </Button>
        <Text variant="small" as="p" tone="muted" className="max-w-md">
          Paste this into the chat so the agent can regenerate a working
          version.
        </Text>
      </div>
    );
  }
}
