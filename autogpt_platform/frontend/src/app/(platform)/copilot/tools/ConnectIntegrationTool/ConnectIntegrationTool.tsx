"use client";

import type { SetupRequirementsResponse } from "@/app/api/__generated__/models/setupRequirementsResponse";
import type { ToolUIPart } from "ai";
import { MorphingTextAnimation } from "../../components/MorphingTextAnimation/MorphingTextAnimation";
import { SetupRequirementsCard } from "../RunBlock/components/SetupRequirementsCard/SetupRequirementsCard";

type Props = {
  part: ToolUIPart;
};

function parseOutput(raw: unknown): SetupRequirementsResponse | null {
  try {
    let parsed: unknown = raw;
    if (typeof raw === "string") {
      parsed = JSON.parse(raw);
    }
    if (parsed && typeof parsed === "object" && "setup_info" in parsed) {
      return parsed as SetupRequirementsResponse;
    }
  } catch {
    // ignore parse errors
  }
  return null;
}

function parseError(raw: unknown): string | null {
  try {
    let parsed: unknown = raw;
    if (typeof raw === "string") {
      parsed = JSON.parse(raw);
    }
    if (parsed && typeof parsed === "object" && "message" in parsed) {
      return String((parsed as { message: unknown }).message);
    }
  } catch {
    // ignore parse errors
  }
  return null;
}

export function ConnectIntegrationTool({ part }: Props) {
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

  const providerName =
    output?.setup_info?.agent_name ??
    (part as { input?: { provider?: string } }).input?.provider ??
    "integration";

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
          <SetupRequirementsCard
            output={output}
            credentialsLabel={`${output.setup_info?.agent_name ?? providerName} credentials`}
            retryInstruction="I've connected my account. Please continue."
          />
        </div>
      )}
    </div>
  );
}
