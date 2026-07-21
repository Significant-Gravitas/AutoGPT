import type { ErrorResponse } from "@/app/api/__generated__/models/errorResponse";
import type { NeedLoginResponse } from "@/app/api/__generated__/models/needLoginResponse";
import { ResponseType } from "@/app/api/__generated__/models/responseType";
import type { SetupRequirementsResponse } from "@/app/api/__generated__/models/setupRequirementsResponse";
import { WarningDiamondIcon, WebhooksLogoIcon } from "@phosphor-icons/react";
import type { ToolUIPart } from "ai";
import { ScaleLoader } from "../../components/ScaleLoader/ScaleLoader";

/** Success output of the `setup_agent_webhook_trigger` tool. Duck-typed: it's a
 *  copilot tool result, not a REST response model. */
export interface TriggerSetupOutput {
  type?: string;
  message: string;
  preset_id: string;
  library_agent_id: string;
  library_agent_link: string;
  name: string;
  is_active: boolean;
  provider: string;
  manual_setup_required: boolean;
  webhook_url?: string | null;
}

/** Returned when the trigger block needs config the user must provide (e.g. a
 *  GitHub repo + event filter). The LLM asks the user and re-calls; this is
 *  duck-typed since it's a copilot tool result, not a REST response model. */
export interface TriggerConfigRequiredOutput {
  type?: string;
  message: string;
  missing_config: string[];
  config_schema: Record<string, unknown>;
}

export type SetupTriggerToolOutput =
  | SetupRequirementsResponse
  | TriggerSetupOutput
  | TriggerConfigRequiredOutput
  | NeedLoginResponse
  | ErrorResponse;

export function isSetupTriggerSetupRequirementsOutput(
  output: SetupTriggerToolOutput,
): output is SetupRequirementsResponse {
  return (
    output.type === ResponseType.setup_requirements ||
    ("setup_info" in output && typeof output.setup_info === "object")
  );
}

export function isTriggerSetupOutput(
  output: SetupTriggerToolOutput,
): output is TriggerSetupOutput {
  return (
    output.type === ResponseType.trigger_setup ||
    "manual_setup_required" in output
  );
}

export function isTriggerConfigRequiredOutput(
  output: SetupTriggerToolOutput,
): output is TriggerConfigRequiredOutput {
  return (
    output.type === ResponseType.trigger_config_required ||
    "missing_config" in output
  );
}

export function isSetupTriggerNeedLoginOutput(
  output: SetupTriggerToolOutput,
): output is NeedLoginResponse {
  return output.type === ResponseType.need_login;
}

export function isSetupTriggerErrorOutput(
  output: SetupTriggerToolOutput,
): output is ErrorResponse {
  return output.type === ResponseType.error || "error" in output;
}

function parseOutput(output: unknown): SetupTriggerToolOutput | null {
  if (!output) return null;
  if (typeof output === "string") {
    const trimmed = output.trim();
    if (!trimmed) return null;
    try {
      return parseOutput(JSON.parse(trimmed) as unknown);
    } catch {
      return null;
    }
  }
  if (typeof output === "object") {
    if ("setup_info" in output) return output as SetupRequirementsResponse;
    if ("manual_setup_required" in output) return output as TriggerSetupOutput;
    if ("missing_config" in output)
      return output as TriggerConfigRequiredOutput;
    if ("error" in output || "details" in output)
      return output as ErrorResponse;
    if ((output as { type?: unknown }).type === ResponseType.need_login)
      return output as NeedLoginResponse;
  }
  return null;
}

export function getSetupTriggerToolOutput(
  part: unknown,
): SetupTriggerToolOutput | null {
  if (!part || typeof part !== "object") return null;
  return parseOutput((part as { output?: unknown }).output);
}

export function getAnimationText(part: {
  state: ToolUIPart["state"];
  output?: unknown;
}): string {
  switch (part.state) {
    case "input-streaming":
    case "input-available":
      return "Setting up the trigger";
    case "output-available": {
      const output = parseOutput(part.output);
      if (!output) return "Setting up the trigger";
      if (isTriggerSetupOutput(output))
        return `Trigger "${output.name}" set up`;
      if (isSetupTriggerSetupRequirementsOutput(output))
        return `Setup needed for "${output.setup_info.agent_name}"`;
      if (isTriggerConfigRequiredOutput(output))
        return "Trigger configuration needed";
      if (isSetupTriggerNeedLoginOutput(output))
        return "Sign in required to set up the trigger";
      return "Something went wrong";
    }
    case "output-error":
      return "Something went wrong";
    default:
      return "Setting up the trigger";
  }
}

export function ToolIcon({
  isStreaming,
  isError,
}: {
  isStreaming?: boolean;
  isError?: boolean;
}) {
  if (isError) {
    return (
      <WarningDiamondIcon size={14} weight="regular" className="text-red-500" />
    );
  }
  if (isStreaming) {
    return <ScaleLoader size={14} />;
  }
  return (
    <WebhooksLogoIcon size={14} weight="regular" className="text-neutral-400" />
  );
}
