"use client";

import type { SetupRequirementsResponse } from "@/app/api/__generated__/models/setupRequirementsResponse";
import type { ToolUIPart } from "ai";
import { useState } from "react";
import { MorphingTextAnimation } from "../../components/MorphingTextAnimation/MorphingTextAnimation";
import { ContentMessage } from "../../components/ToolAccordion/AccordionContent";
import { SetupRequirementsCard } from "../RunBlock/components/SetupRequirementsCard/SetupRequirementsCard";

type Props = {
  part: ToolUIPart;
};

function parseJson(raw: unknown): unknown {
  if (typeof raw === "string") {
    try {
      return JSON.parse(raw);
    } catch {
      return null;
    }
  }
  return raw;
}

function parseOutput(raw: unknown): SetupRequirementsResponse | null {
  const parsed = parseJson(raw);
  if (parsed && typeof parsed === "object" && "setup_info" in parsed) {
    return parsed as SetupRequirementsResponse;
  }
  return null;
}

function parseError(raw: unknown): string | null {
  const parsed = parseJson(raw);
  if (parsed && typeof parsed === "object" && "message" in parsed) {
    return String((parsed as { message: unknown }).message);
  }
  return null;
}

export function ConnectIntegrationTool({ part }: Props) {
  // Persist dismissed state here so SetupRequirementsCard remounts don't re-enable Proceed.
  const [isDismissed, setIsDismissed] = useState(false);

  const isStreaming =
    part.state === "input-streaming" || part.state === "input-available";
  const isError = part.state === "output-error";

  const output =
    part.state === "output-available"
      ? parseOutput((part as { output?: unknown }).output)
      : null;

  const errorMessage = isError
    ? (parseError((part as { output?: unknown }).output) ??
      "Failed to connect integration")
    : null;

  const rawProvider =
    (part as { input?: { provider?: string } }).input?.provider ?? "";
  const providerName =
    output?.setup_info?.agent_name ??
    // Sanitize LLM-controlled provider slug: trim and cap at 64 chars to
    // prevent runaway text in the DOM.
    (rawProvider ? rawProvider.trim().slice(0, 64) : "integration");

  const label = isStreaming
    ? `Connecting ${providerName}…`
    : isError
      ? `Failed to connect ${providerName}`
      : output
        ? `Connect ${output.setup_info?.agent_name ?? providerName}`
        : `Connect ${providerName}`;

  return (
    <div className="py-2">
      <div className="flex items-center gap-2 text-sm text-muted-foreground">
        <MorphingTextAnimation
          text={label}
          className={isError ? "text-red-500" : undefined}
        />
      </div>

      {isError && errorMessage && (
        <p className="mt-1 text-sm text-red-500">{errorMessage}</p>
      )}

      {output && (
        <div className="mt-2">
          {isDismissed ? (
            <ContentMessage>Connected. Continuing…</ContentMessage>
          ) : (
            <SetupRequirementsCard
              output={output}
              credentialsLabel={`${output.setup_info?.agent_name ?? providerName} credentials`}
              retryInstruction="I've connected my account. Please continue."
              onComplete={() => setIsDismissed(true)}
            />
          )}
        </div>
      )}
    </div>
  );
}
