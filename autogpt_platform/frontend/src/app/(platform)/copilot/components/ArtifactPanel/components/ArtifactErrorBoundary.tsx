"use client";

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
        <p className="text-sm font-medium text-zinc-700">
          This artifact couldn&apos;t be rendered
        </p>
        <p className="max-w-md break-words text-xs text-zinc-500">
          Something in{" "}
          <span className="font-mono">{this.props.artifactTitle}</span> threw an
          error while rendering. The chat and sidebar are still working.
        </p>
        <pre className="max-h-32 max-w-md overflow-auto whitespace-pre-wrap break-words rounded-md bg-zinc-100 px-3 py-2 text-left text-xs text-zinc-700">
          {message}
        </pre>
        <button
          type="button"
          onClick={this.handleCopy}
          className="rounded-md border border-zinc-200 bg-white px-3 py-1.5 text-xs font-medium text-zinc-700 shadow-sm transition-colors hover:bg-zinc-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-400"
        >
          Copy error details
        </button>
        <p className="max-w-md text-xs text-zinc-400">
          Paste this into the chat so the agent can regenerate a working
          version.
        </p>
      </div>
    );
  }
}
