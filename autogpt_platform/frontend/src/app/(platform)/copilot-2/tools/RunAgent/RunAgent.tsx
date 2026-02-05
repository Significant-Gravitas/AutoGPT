"use client";

import type { ToolUIPart } from "ai";
import Link from "next/link";
import { MorphingTextAnimation } from "../../components/MorphingTextAnimation/MorphingTextAnimation";
import { ToolAccordion } from "../../components/ToolAccordion/ToolAccordion";
import { useCopilotChatActions } from "../../components/CopilotChatActionsProvider/useCopilotChatActions";
import {
  ChatCredentialsSetup,
  type CredentialInfo,
} from "@/components/contextual/Chat/components/ChatCredentialsSetup/ChatCredentialsSetup";
import {
  formatMaybeJson,
  getAnimationText,
  getRunAgentToolOutput,
  isRunAgentAgentDetailsOutput,
  isRunAgentErrorOutput,
  isRunAgentExecutionStartedOutput,
  isRunAgentNeedLoginOutput,
  isRunAgentSetupRequirementsOutput,
  ToolIcon,
  type RunAgentToolOutput,
} from "./helpers";

export interface RunAgentToolPart {
  type: string;
  toolCallId: string;
  state: ToolUIPart["state"];
  input?: unknown;
  output?: unknown;
}

interface Props {
  part: RunAgentToolPart;
}

function getAccordionMeta(output: RunAgentToolOutput): {
  badgeText: string;
  title: string;
  description?: string;
} {
  if (isRunAgentExecutionStartedOutput(output)) {
    const statusText =
      typeof output.status === "string" && output.status.trim()
        ? output.status.trim()
        : "started";
    return {
      badgeText: "Run agent",
      title: output.graph_name,
      description: `Status: ${statusText}`,
    };
  }

  if (isRunAgentAgentDetailsOutput(output)) {
    return {
      badgeText: "Run agent",
      title: output.agent.name,
      description: "Inputs required",
    };
  }

  if (isRunAgentSetupRequirementsOutput(output)) {
    const missingCredsCount = Object.keys(
      (output.setup_info.user_readiness?.missing_credentials ?? {}) as Record<
        string,
        unknown
      >,
    ).length;
    return {
      badgeText: "Run agent",
      title: output.setup_info.agent_name,
      description:
        missingCredsCount > 0
          ? `Missing ${missingCredsCount} credential${missingCredsCount === 1 ? "" : "s"}`
          : output.message,
    };
  }

  if (isRunAgentNeedLoginOutput(output)) {
    return { badgeText: "Run agent", title: "Sign in required" };
  }

  return { badgeText: "Run agent", title: "Error" };
}

function coerceMissingCredentials(
  rawMissingCredentials: unknown,
): CredentialInfo[] {
  const missing =
    rawMissingCredentials && typeof rawMissingCredentials === "object"
      ? (rawMissingCredentials as Record<string, unknown>)
      : {};

  const validTypes = new Set([
    "api_key",
    "oauth2",
    "user_password",
    "host_scoped",
  ]);

  const results: CredentialInfo[] = [];

  Object.values(missing).forEach((value) => {
    if (!value || typeof value !== "object") return;
    const cred = value as Record<string, unknown>;

    const provider =
      typeof cred.provider === "string" ? cred.provider.trim() : "";
    if (!provider) return;

    const providerName =
      typeof cred.provider_name === "string" && cred.provider_name.trim()
        ? cred.provider_name.trim()
        : provider.replace(/_/g, " ");

    const title =
      typeof cred.title === "string" && cred.title.trim()
        ? cred.title.trim()
        : providerName;

    const types =
      Array.isArray(cred.types) && cred.types.length > 0
        ? cred.types
        : typeof cred.type === "string"
          ? [cred.type]
          : [];

    const credentialTypes = types
      .map((t) => (typeof t === "string" ? t.trim() : ""))
      .filter(
        (t): t is "api_key" | "oauth2" | "user_password" | "host_scoped" =>
          validTypes.has(t),
      );

    if (credentialTypes.length === 0) return;

    const scopes = Array.isArray(cred.scopes)
      ? cred.scopes.filter((s): s is string => typeof s === "string")
      : undefined;

    const item: CredentialInfo = {
      provider,
      providerName,
      credentialTypes,
      title,
    };
    if (scopes && scopes.length > 0) {
      item.scopes = scopes;
    }
    results.push(item);
  });

  return results;
}

function coerceExpectedInputs(rawInputs: unknown): Array<{
  name: string;
  title: string;
  type: string;
  description?: string;
  required: boolean;
}> {
  if (!Array.isArray(rawInputs)) return [];
  const results: Array<{
    name: string;
    title: string;
    type: string;
    description?: string;
    required: boolean;
  }> = [];

  rawInputs.forEach((value, index) => {
    if (!value || typeof value !== "object") return;
    const input = value as Record<string, unknown>;

    const name =
      typeof input.name === "string" && input.name.trim()
        ? input.name.trim()
        : `input-${index}`;
    const title =
      typeof input.title === "string" && input.title.trim()
        ? input.title.trim()
        : name;
    const type = typeof input.type === "string" ? input.type : "unknown";
    const description =
      typeof input.description === "string" && input.description.trim()
        ? input.description.trim()
        : undefined;
    const required = Boolean(input.required);

    const item: {
      name: string;
      title: string;
      type: string;
      description?: string;
      required: boolean;
    } = { name, title, type, required };
    if (description) item.description = description;
    results.push(item);
  });

  return results;
}

export function RunAgentTool({ part }: Props) {
  const text = getAnimationText(part);
  const { onSend } = useCopilotChatActions();
  const isStreaming =
    part.state === "input-streaming" || part.state === "input-available";

  const output = getRunAgentToolOutput(part);
  const isError =
    part.state === "output-error" || (!!output && isRunAgentErrorOutput(output));
  const hasExpandableContent =
    part.state === "output-available" &&
    !!output &&
    (isRunAgentExecutionStartedOutput(output) ||
      isRunAgentAgentDetailsOutput(output) ||
      isRunAgentSetupRequirementsOutput(output) ||
      isRunAgentNeedLoginOutput(output) ||
      isRunAgentErrorOutput(output));

  function handleAllCredentialsComplete() {
    onSend(
      "I've configured the required credentials. Please check if everything is ready and proceed with running the agent.",
    );
  }

  return (
    <div className="py-2">
      <div className="flex items-center gap-2 text-sm text-muted-foreground">
        <ToolIcon isStreaming={isStreaming} isError={isError} />
        <MorphingTextAnimation
          text={text}
          className={isError ? "text-red-500" : undefined}
        />
      </div>

      {hasExpandableContent && output && (
        <ToolAccordion
          {...getAccordionMeta(output)}
          defaultExpanded={
            isRunAgentSetupRequirementsOutput(output) ||
            isRunAgentAgentDetailsOutput(output)
          }
        >
          {isRunAgentExecutionStartedOutput(output) && (
            <div className="grid gap-2">
              <div className="rounded-2xl border bg-background p-3">
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0">
                    <p className="text-sm font-medium text-foreground">
                      Execution started
                    </p>
                    <p className="mt-0.5 truncate text-xs text-muted-foreground">
                      {output.execution_id}
                    </p>
                    <p className="mt-2 text-xs text-muted-foreground">
                      {output.message}
                    </p>
                  </div>
                  {output.library_agent_link && (
                    <Link
                      href={output.library_agent_link}
                      className="shrink-0 text-xs font-medium text-purple-600 hover:text-purple-700"
                    >
                      Open
                    </Link>
                  )}
                </div>
              </div>
            </div>
          )}

          {isRunAgentAgentDetailsOutput(output) && (
            <div className="grid gap-2">
              <p className="text-sm text-foreground">{output.message}</p>

              {output.agent.description?.trim() && (
                <p className="text-xs text-muted-foreground">
                  {output.agent.description}
                </p>
              )}

              <div className="rounded-2xl border bg-background p-3">
                <p className="text-xs font-medium text-foreground">Inputs</p>
                <p className="mt-1 text-xs text-muted-foreground">
                  Provide required inputs in chat, or ask to run with defaults.
                </p>
                <pre className="mt-2 whitespace-pre-wrap text-xs text-muted-foreground">
                  {formatMaybeJson(output.agent.inputs)}
                </pre>
              </div>
            </div>
          )}

          {isRunAgentSetupRequirementsOutput(output) && (
            <div className="grid gap-2">
              <p className="text-sm text-foreground">{output.message}</p>

              {coerceMissingCredentials(
                output.setup_info.user_readiness?.missing_credentials,
              ).length > 0 && (
                <ChatCredentialsSetup
                  credentials={coerceMissingCredentials(
                    output.setup_info.user_readiness?.missing_credentials,
                  )}
                  agentName={output.setup_info.agent_name}
                  message={output.message}
                  onAllCredentialsComplete={handleAllCredentialsComplete}
                  onCancel={() => {}}
                />
              )}

              {coerceExpectedInputs(
                (output.setup_info.requirements as Record<string, unknown>)
                  ?.inputs,
              ).length > 0 && (
                <div className="rounded-2xl border bg-background p-3">
                  <p className="text-xs font-medium text-foreground">
                    Expected inputs
                  </p>
                  <div className="mt-2 grid gap-2">
                    {coerceExpectedInputs(
                      (
                        output.setup_info.requirements as Record<
                          string,
                          unknown
                        >
                      )?.inputs,
                    ).map((input) => (
                      <div key={input.name} className="rounded-xl border p-2">
                        <div className="flex items-center justify-between gap-2">
                          <p className="truncate text-xs font-medium text-foreground">
                            {input.title}
                          </p>
                          <span className="shrink-0 rounded-full border bg-muted px-2 py-0.5 text-[11px] text-muted-foreground">
                            {input.required ? "Required" : "Optional"}
                          </span>
                        </div>
                        <p className="mt-1 text-xs text-muted-foreground">
                          {input.name} • {input.type}
                          {input.description ? ` • ${input.description}` : ""}
                        </p>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {isRunAgentNeedLoginOutput(output) && (
            <p className="text-sm text-foreground">{output.message}</p>
          )}

          {isRunAgentErrorOutput(output) && (
            <div className="grid gap-2">
              <p className="text-sm text-foreground">{output.message}</p>
              {output.error && (
                <pre className="whitespace-pre-wrap rounded-2xl border bg-muted/30 p-3 text-xs text-muted-foreground">
                  {formatMaybeJson(output.error)}
                </pre>
              )}
              {output.details && (
                <pre className="whitespace-pre-wrap rounded-2xl border bg-muted/30 p-3 text-xs text-muted-foreground">
                  {formatMaybeJson(output.details)}
                </pre>
              )}
            </div>
          )}
        </ToolAccordion>
      )}
    </div>
  );
}
