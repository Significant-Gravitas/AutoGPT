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

export function ConnectIntegrationTool({ part }: Props) {
  const isStreaming =
    part.state === "input-streaming" || part.state === "input-available";

  const output =
    part.state === "output-available"
      ? parseOutput((part as { output?: unknown }).output)
      : null;

  const providerName =
    output?.setup_info?.agent_name ??
    (part as { input?: { provider?: string } }).input?.provider ??
    "integration";

  const label = isStreaming
    ? `Connecting ${providerName}…`
    : output
      ? `Connect ${output.setup_info.agent_name}`
      : `Connect ${providerName}`;

  return (
    <div className="py-2">
      <div className="flex items-center gap-2 text-sm text-muted-foreground">
        <MorphingTextAnimation text={label} />
      </div>

      {output && (
        <div className="mt-2">
          <SetupRequirementsCard
            output={output}
            credentialsLabel="Integration credentials"
            retryInstruction="I've connected my account. Please continue."
          />
        </div>
      )}
    </div>
  );
}
