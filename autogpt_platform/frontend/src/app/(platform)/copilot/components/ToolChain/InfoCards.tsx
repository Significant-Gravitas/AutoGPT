"use client";

import {
  AlertCircleIcon,
  BookBookmarkIcon,
  CheckmarkCircle02Icon,
  CircleIcon,
  ConnectIcon,
  MessageQuestionIcon,
} from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import { useContext } from "react";
import { CopilotChatActionsContext } from "../CopilotChatActionsProvider/useCopilotChatActions";
import { CardProviderIcon } from "./BlockCards";
import { CARD, HALF, StatusCard } from "./ResultCards";
import { asObject, inline, resultItemKey, str } from "./resultHelpers";

interface ItemsProps {
  steps: Record<string, unknown>[];
}

interface OutputProps {
  output: Record<string, unknown>;
}

interface ErrorListProps {
  errors: string[];
}

interface QuestionsCardProps {
  questions: Record<string, unknown>[];
}

interface SetupCardProps extends OutputProps {
  provider: string | null;
}

export function PlanSteps({ steps }: ItemsProps) {
  return (
    <div className={CARD + " flex flex-col gap-1.5 p-3"}>
      {steps.map((step, i) => {
        const status = str(step, "status");
        const done = status === "completed";
        const active = status === "in_progress";
        const blockName = str(step, "block_name");
        return (
          <div
            key={resultItemKey(step, i)}
            className="flex items-center gap-2 text-[13px]"
          >
            {done ? (
              <Icon
                icon={CheckmarkCircle02Icon}
                size={15}
                className="shrink-0 text-green-500"
              />
            ) : (
              <Icon
                icon={CircleIcon}
                size={15}
                className={
                  "shrink-0 " + (active ? "text-purple-500" : "text-zinc-300")
                }
              />
            )}
            <span
              className={
                "min-w-0 flex-1 truncate " +
                (done
                  ? "text-zinc-400 line-through"
                  : active
                    ? "font-medium text-zinc-800"
                    : "text-zinc-600")
              }
            >
              {str(step, "description") ?? inline(step)}
            </span>
            {blockName && (
              <span className="shrink-0 rounded-full bg-zinc-100 px-2 py-0.5 text-[11px] text-zinc-500">
                {blockName}
              </span>
            )}
          </div>
        );
      })}
    </div>
  );
}

function ErrorList({ errors }: ErrorListProps) {
  return (
    <div className={CARD + " flex flex-col gap-1.5 p-2.5"}>
      {errors.map((error) => (
        <div
          key={error}
          className="flex items-start gap-2 text-xs text-zinc-600"
        >
          <Icon
            icon={AlertCircleIcon}
            size={14}
            className="mt-px shrink-0 text-red-400"
          />
          <span className="min-w-0 break-words">{error}</span>
        </div>
      ))}
    </div>
  );
}

function strList(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter((item): item is string => typeof item === "string")
    : [];
}

export function ValidationCard({ output }: OutputProps) {
  const errors = strList(output.errors);
  if (output.valid === true) return <StatusCard ok label="Graph is valid" />;
  if (errors.length > 0) return <ErrorList errors={errors} />;
  return <StatusCard ok={false} label="Graph has errors" />;
}

export function FixResultCard({ output }: OutputProps) {
  const fixes = strList(output.fixes_applied);
  const remaining = strList(output.remaining_errors);
  return (
    <div className="flex flex-col gap-1.5">
      <StatusCard
        ok={output.valid_after_fix === true}
        label={
          output.valid_after_fix === true
            ? `Fixed — applied ${fixes.length} fix${fixes.length === 1 ? "" : "es"}`
            : `${remaining.length} error${remaining.length === 1 ? "" : "s"} remaining`
        }
      />
      {remaining.length > 0 && <ErrorList errors={remaining} />}
    </div>
  );
}

export function QuestionsCard({ questions }: QuestionsCardProps) {
  return (
    <div className={`${CARD} ${HALF} flex flex-col gap-2 p-2.5`}>
      {questions.map((entry, i) => (
        <div key={resultItemKey(entry, i)} className="flex items-start gap-2">
          <Icon
            icon={MessageQuestionIcon}
            size={14}
            className="mt-0.5 shrink-0 text-zinc-400"
          />
          <div className="min-w-0">
            <p className="text-[13px] text-zinc-700">
              {str(entry, "question") ?? inline(entry)}
            </p>
            {str(entry, "example") && (
              <p className="truncate text-xs text-zinc-400">
                e.g. {str(entry, "example")}
              </p>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}

export function SetupCard({ output, provider }: SetupCardProps) {
  const setupInfo = asObject(output.setup_info);
  const name = setupInfo ? str(setupInfo, "agent_name") : null;
  if (!name) return null;
  return (
    <div className={`${CARD} ${HALF} flex items-center gap-2.5 p-2.5`}>
      <div className="flex size-7 shrink-0 items-center justify-center rounded-full bg-zinc-100">
        <CardProviderIcon
          provider={provider}
          fallback={
            <Icon icon={ConnectIcon} size={15} className="text-zinc-600" />
          }
        />
      </div>
      <div className="min-w-0 flex-1">
        <p className="truncate text-[13px] font-medium text-zinc-800">{name}</p>
        <p className="truncate text-xs text-zinc-500">Connection required</p>
      </div>
    </div>
  );
}

export function SkillCard({ output }: OutputProps) {
  const name = str(output, "name");
  if (!name) return null;
  const triggers = strList(output.triggers);
  return (
    <div className={`${CARD} ${HALF} flex items-center gap-2.5 p-2.5`}>
      <div className="flex size-7 shrink-0 items-center justify-center rounded-full bg-zinc-100">
        <Icon icon={BookBookmarkIcon} size={15} className="text-zinc-600" />
      </div>
      <div className="min-w-0 flex-1">
        <p className="truncate text-[13px] font-medium text-zinc-800">{name}</p>
        {str(output, "description") && (
          <p className="truncate text-xs text-zinc-500">
            {str(output, "description")}
          </p>
        )}
      </div>
      {triggers[0] && (
        <span className="shrink-0 rounded-full bg-zinc-100 px-2 py-0.5 text-[11px] text-zinc-500">
          {triggers[0]}
        </span>
      )}
    </div>
  );
}

export function SuggestedGoalCard({ output }: OutputProps) {
  const actions = useContext(CopilotChatActionsContext);
  const goal = str(output, "suggested_goal");
  if (!goal) return null;

  return (
    <div className={`${CARD} ${HALF} flex flex-col gap-2 p-3`}>
      <p className="text-[13px] font-medium text-zinc-800">{goal}</p>
      {str(output, "reason") && (
        <p className="text-xs text-zinc-500">{str(output, "reason")}</p>
      )}
      {actions && (
        <button
          type="button"
          className="w-fit rounded-lg bg-zinc-900 px-2.5 py-1.5 text-xs font-medium text-white transition-colors hover:bg-zinc-700"
          onClick={() =>
            actions.onSend(`Please create an agent with this goal: ${goal}`)
          }
        >
          Use this goal
        </button>
      )}
    </div>
  );
}

export function TriggerSetupCard({ output }: OutputProps) {
  const url = str(output, "webhook_url", "ingress_url");

  async function handleCopy() {
    if (!url) return;
    try {
      await navigator.clipboard.writeText(url);
    } catch {
      return;
    }
  }

  return (
    <div className={`${CARD} ${HALF} flex flex-col gap-2 p-3`}>
      <p className="text-[13px] text-zinc-700">
        {str(output, "message", "name") ?? "Trigger is ready"}
      </p>
      {url && (
        <div className="flex items-center gap-2 rounded-lg bg-zinc-50 p-2 ring-1 ring-zinc-200/70">
          <code className="min-w-0 flex-1 break-all text-xs text-zinc-600">
            {url}
          </code>
          <button
            type="button"
            onClick={handleCopy}
            className="shrink-0 rounded-md px-2 py-1 text-xs font-medium text-zinc-600 transition-colors hover:bg-zinc-200"
          >
            Copy
          </button>
        </div>
      )}
    </div>
  );
}
