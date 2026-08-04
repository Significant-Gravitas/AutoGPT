"use client";

import { Button } from "@/components/atoms/Button/Button";
import { CheckIcon, CopyIcon } from "@phosphor-icons/react";
import type { ToolUIPart } from "ai";
import { useState } from "react";
import { MorphingTextAnimation } from "../../components/MorphingTextAnimation/MorphingTextAnimation";
import { SetupRequirementsCard } from "../../components/SetupRequirementsCard/SetupRequirementsCard";
import { ContentMessage } from "../../components/ToolAccordion/AccordionContent";
import {
  getAnimationText,
  getSetupTriggerToolOutput,
  isSetupTriggerErrorOutput,
  isSetupTriggerNeedLoginOutput,
  isSetupTriggerSetupRequirementsOutput,
  isTriggerConfigRequiredOutput,
  isTriggerSetupOutput,
  ToolIcon,
  type TriggerSetupOutput,
} from "./helpers";

interface Props {
  part: {
    type: string;
    state: ToolUIPart["state"];
    input?: unknown;
    output?: unknown;
  };
}

export function SetupTriggerTool({ part }: Props) {
  const text = getAnimationText(part);
  const isStreaming =
    part.state === "input-streaming" || part.state === "input-available";

  const output = getSetupTriggerToolOutput(part);
  const isOutputAvailable = part.state === "output-available" && !!output;

  const setupRequirementsOutput =
    isOutputAvailable && isSetupTriggerSetupRequirementsOutput(output)
      ? output
      : null;
  const successOutput =
    isOutputAvailable &&
    !setupRequirementsOutput &&
    isTriggerSetupOutput(output)
      ? output
      : null;
  const configRequiredOutput =
    isOutputAvailable && isTriggerConfigRequiredOutput(output) ? output : null;
  const needLoginOutput =
    isOutputAvailable && isSetupTriggerNeedLoginOutput(output) ? output : null;
  const isError =
    part.state === "output-error" ||
    (isOutputAvailable && isSetupTriggerErrorOutput(output));

  return (
    <div className="py-2">
      {!setupRequirementsOutput &&
        !successOutput &&
        !configRequiredOutput &&
        !needLoginOutput && (
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <ToolIcon isStreaming={isStreaming} isError={isError} />
            <MorphingTextAnimation
              text={text}
              className={isError ? "text-red-500" : undefined}
            />
          </div>
        )}

      {setupRequirementsOutput && (
        <div className="mt-2">
          <SetupRequirementsCard
            output={setupRequirementsOutput}
            inputsMode="trigger"
            credentialsLabel="Account"
          />
        </div>
      )}

      {successOutput && (
        <div className="mt-2">
          <TriggerSetupSuccessCard output={successOutput} />
        </div>
      )}

      {configRequiredOutput && (
        <div className="mt-2">
          <ContentMessage>{configRequiredOutput.message}</ContentMessage>
        </div>
      )}

      {needLoginOutput && (
        <div className="mt-2">
          <ContentMessage>{needLoginOutput.message}</ContentMessage>
        </div>
      )}
    </div>
  );
}

function TriggerSetupSuccessCard({ output }: { output: TriggerSetupOutput }) {
  const [copied, setCopied] = useState(false);

  async function handleCopy() {
    if (!output.webhook_url) return;
    try {
      await navigator.clipboard.writeText(output.webhook_url);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Clipboard API unavailable (e.g. non-secure context) — the URL stays
      // visible in the card for manual copy, so just skip the confirmation.
    }
  }

  return (
    <div className="grid gap-2 rounded-2xl border bg-background p-3">
      <ContentMessage>{output.message}</ContentMessage>
      {output.manual_setup_required && output.webhook_url && (
        <div className="flex items-center gap-2 rounded-xl border bg-muted/40 p-2">
          <code className="min-w-0 flex-1 break-all text-xs">
            {output.webhook_url}
          </code>
          <Button
            variant="ghost"
            size="small"
            onClick={handleCopy}
            aria-label={copied ? "Copied" : "Copy webhook URL"}
          >
            {copied ? <CheckIcon size={14} /> : <CopyIcon size={14} />}
          </Button>
        </div>
      )}
    </div>
  );
}
