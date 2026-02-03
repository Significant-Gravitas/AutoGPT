"use client";

import type { ToolUIPart } from "ai";
import Link from "next/link";
import { MorphingTextAnimation } from "../../components/MorphingTextAnimation/MorphingTextAnimation";
import { ToolAccordion } from "../../components/ToolAccordion/ToolAccordion";
import { useCopilotChatActions } from "../../components/CopilotChatActionsProvider/useCopilotChatActions";
import { ChatCredentialsSetup } from "@/components/contextual/Chat/components/ChatCredentialsSetup/ChatCredentialsSetup";
import {
  formatMaybeJson,
  getAnimationText,
  getRunAgentToolOutput,
  StateIcon,
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
  if (output.type === "execution_started") {
    return {
      badgeText: "Run agent",
      title: output.graph_name,
      description: `Status: ${output.status}`,
    };
  }

  if (output.type === "agent_details") {
    return {
      badgeText: "Run agent",
      title: output.agent.name,
      description: "Inputs required",
    };
  }

  if (output.type === "setup_requirements") {
    const missingCredsCount = Object.keys(
      output.setup_info.user_readiness.missing_credentials ?? {},
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

  if (output.type === "need_login") {
    return { badgeText: "Run agent", title: "Sign in required" };
  }

  return { badgeText: "Run agent", title: "Error" };
}

export function RunAgentTool({ part }: Props) {
  const text = getAnimationText(part);
  const { onSend } = useCopilotChatActions();

  const output = getRunAgentToolOutput(part);
  const hasExpandableContent =
    part.state === "output-available" &&
    !!output &&
    (output.type === "execution_started" ||
      output.type === "agent_details" ||
      output.type === "setup_requirements" ||
      output.type === "need_login" ||
      output.type === "error");

  function handleAllCredentialsComplete() {
    onSend(
      "I've configured the required credentials. Please check if everything is ready and proceed with running the agent.",
    );
  }

  return (
    <div className="py-2">
      <div className="flex items-center gap-2 text-sm text-muted-foreground">
        <StateIcon state={part.state} />
        <MorphingTextAnimation text={text} />
      </div>

      {hasExpandableContent && output && (
        <ToolAccordion
          {...getAccordionMeta(output)}
          defaultExpanded={
            output.type === "setup_requirements" ||
            output.type === "agent_details"
          }
        >
          {output.type === "execution_started" && (
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

          {output.type === "agent_details" && (
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

          {output.type === "setup_requirements" && (
            <div className="grid gap-2">
              <p className="text-sm text-foreground">{output.message}</p>

              {Object.keys(
                output.setup_info.user_readiness.missing_credentials ?? {},
              ).length > 0 && (
                <ChatCredentialsSetup
                  credentials={Object.values(
                    output.setup_info.user_readiness.missing_credentials ?? {},
                  ).map((cred) => ({
                    provider: cred.provider,
                    providerName:
                      cred.provider_name ?? cred.provider.replace(/_/g, " "),
                    credentialTypes: (cred.types ?? [cred.type]) as Array<
                      "api_key" | "oauth2" | "user_password" | "host_scoped"
                    >,
                    title: cred.title,
                    scopes: cred.scopes,
                  }))}
                  agentName={output.setup_info.agent_name}
                  message={output.message}
                  onAllCredentialsComplete={handleAllCredentialsComplete}
                  onCancel={() => {}}
                />
              )}

              {output.setup_info.requirements.inputs?.length > 0 && (
                <div className="rounded-2xl border bg-background p-3">
                  <p className="text-xs font-medium text-foreground">
                    Expected inputs
                  </p>
                  <div className="mt-2 grid gap-2">
                    {output.setup_info.requirements.inputs.map((input) => (
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

          {output.type === "need_login" && (
            <p className="text-sm text-foreground">{output.message}</p>
          )}

          {output.type === "error" && (
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
