"use client";

import {
  ArrowLeft01Icon,
  ArrowRight01Icon,
  InformationCircleIcon,
  PencilEdit02Icon,
  SentIcon,
  UndoIcon,
  Tick02Icon,
  UserAdd01Icon,
} from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";
import type { ToolUIPart } from "ai";
import { type ReactNode, useContext, useState } from "react";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import { cn } from "@/lib/utils";
import { GenericTool } from "../../tools/GenericTool/GenericTool";
import { type ArtifactRef, useCopilotUIStore } from "../../store";
import { toastSendFailure } from "../../helpers/sendFailureToast";
import { CopilotChatActionsContext } from "../CopilotChatActionsProvider/useCopilotChatActions";
import { SOUL_CHANGE_TOOLS } from "./helpers";
import { useLiveTipMoment } from "@/app/(platform)/components/ContextualTips/useTipMoment";
import { asObject, str } from "./resultHelpers";
import {
  handoffOutcomeOf,
  handoffSubject,
  handoffTaskOf,
} from "./expertHandoff";
import { HandoffProposal, HandoffReceipt } from "./ExpertHandoffParts";
import { HandoffStatus } from "./HandoffStatus";

interface Props {
  output: Record<string, unknown>;
  /** Stable key for the panel's history — the tool call that produced
   *  the card. Without one, Show more expands inline instead. */
  artifactId?: string;
  /** Pager controls when several experts share one card slot. */
  pager?: ReactNode;
  /** The decision already taken on this proposal, if any. */
  decision?: Decision | null;
  /** Offered while the proposal is still open; absent once applied, on a
   *  read-only transcript, or when the output carries no confirmation. */
  onDecide?: (decision: Decision) => void;
  /** Takes the decision back so Approve/Decline can be chosen again. */
  onUndo?: () => void;
  /** The turn is still streaming: answers wait until the proposals are all
   *  in, so a click always does what its button says. */
  waiting?: boolean;
  /** Footer control the group adds once every proposal is decided, or
   *  the sent note after. */
  action?: ReactNode;
  /** Which way the pager just moved — the identity block slides in from
   *  that side while the shell (Details, pager, decision) stays put. */
  stepDirection?: StepDirection;
  /** Sends a chat message (a hand-off that didn't start offers Try again);
   *  absent on a read-only transcript. */
  onSend?: (message: string) => void;
}

export type StepDirection = "forward" | "back";

export type Decision = "approved" | "declined";

// Same shell as the chain's action card (connect / inputs / questions) so
// every "your turn" card in the thread reads as one family.
const SHELL =
  "w-full overflow-hidden rounded-3xl border border-separator bg-surface";
const HEADER = "flex items-center gap-2.5 border-b border-separator px-4 py-3";
const FOOTER =
  "flex items-center justify-between gap-2 border-t border-separator px-4 py-3";

type ExpertKind = "hire" | "raise" | "update";

function kindOf(output: Record<string, unknown>): ExpertKind | null {
  const preview = asObject(output.preview);
  const kind = str(output, "kind") ?? (preview ? str(preview, "kind") : null);
  return kind === "hire" || kind === "raise" || kind === "update" ? kind : null;
}

function headerFor(
  kind: ExpertKind | null,
  applied: boolean,
): { icon: IconSvgElement; title: string } {
  const icon = kind === "update" ? PencilEdit02Icon : UserAdd01Icon;
  if (applied) {
    return {
      icon,
      title:
        kind === "hire"
          ? "Expert hired"
          : kind === "update"
            ? "Expert updated"
            : "Expert created",
    };
  }
  return {
    icon,
    title:
      kind === "hire"
        ? "Hire an expert"
        : kind === "update"
          ? "Update an expert"
          : "Create an Expert",
  };
}

interface Proposal {
  toolCallId: string;
  kind: ExpertKind | "soul" | null;
  name: string;
  confirmationId: string;
  /** The "hire and start" choice, when the hire is for a task. */
  handoff: { confirmationId: string; subject: string; title: string } | null;
}

function isOpenExpertProposal(part: ToolUIPart) {
  const kind = proposalOf(part)?.kind;
  return kind === "hire" || kind === "raise";
}

/** A hire/raise/Soul preview the user can still say yes or no to. */
function proposalOf(part: ToolUIPart): Proposal | null {
  const output = asObject(part.output);
  if (!output || output.applied === true) return null;
  const confirmationId = str(output, "confirmation_id");
  if (confirmationId && isSoulChange(output)) {
    return {
      toolCallId: part.toolCallId,
      kind: "soul",
      name: "your Soul",
      confirmationId,
      handoff: null,
    };
  }
  const preview = asObject(output.preview);
  if (!confirmationId || !preview) return null;
  const name = str(preview, "name") ?? "this expert";
  const task = handoffTaskOf(output);
  const handoffConfirmationId = str(output, "handoff_confirmation_id");
  return {
    toolCallId: part.toolCallId,
    kind: kindOf(output),
    name,
    confirmationId,
    handoff:
      task && handoffConfirmationId
        ? {
            confirmationId: handoffConfirmationId,
            subject: handoffSubject(name, task),
            title: task.title,
          }
        : null,
  };
}

/** The line for one answer. A hire proposed for a task asks one question,
 *  Hire and start or Not now, so its approval is always of the proposal's
 *  hand-off id, which confirm_expert_change reads as "also start the task".
 *  Not now declines the hire itself. */
function answerLine(proposal: Proposal, decision: Decision): string {
  if (decision === "approved" && proposal.handoff) {
    return decisionLine(
      {
        ...proposal,
        name: proposal.handoff.subject,
        confirmationId: proposal.handoff.confirmationId,
      },
      "approved",
    );
  }
  return decisionLine(proposal, decision);
}

// gate/card_approval.py reads the Approved line as the answer to the confirm.
function decisionLine(proposal: Proposal, decision: Decision): string {
  const verb =
    proposal.kind === "hire"
      ? "hire"
      : proposal.kind === "update" || proposal.kind === "soul"
        ? "update"
        : "create";
  return decision === "approved"
    ? `Approved: ${verb} ${proposal.name} (confirmation_id: ${proposal.confirmationId}).`
    : `Not approved: do not ${verb} ${proposal.name}, discard that proposal (confirmation_id: ${proposal.confirmationId}).`;
}

function withoutDecision(
  decisions: Record<string, Decision>,
  toolCallId: string,
) {
  const { [toolCallId]: _, ...rest } = decisions;
  return rest;
}

function toExpertArtifact(
  expert: Record<string, unknown>,
  output: Record<string, unknown>,
  artifactId: string,
): ArtifactRef {
  const name = str(expert, "name") ?? "New expert";
  return {
    id: `expert:${artifactId}`,
    title: name,
    mimeType: null,
    sourceUrl: "",
    origin: "agent",
    expert: {
      id: str(expert, "id") ?? null,
      kind: kindOf(output),
      name,
      role: str(expert, "role") ?? null,
      color: str(expert, "color") ?? null,
      tagline: str(expert, "tagline") ?? null,
      about: str(expert, "about") ?? null,
      boundaries: str(expert, "boundaries") ?? null,
      voicePreferences: str(expert, "voice_preferences") ?? null,
      weeklyBudget:
        typeof expert.weekly_budget === "number" ? expert.weekly_budget : null,
      avatarUrl: str(expert, "avatar_url") ?? null,
      applied: output.applied === true,
    },
  };
}

/** An expert change as its own message part — either a hire/raise preview
 *  awaiting the user's OK (given in chat, nothing created yet) or the
 *  teammate ``confirm_expert_change`` actually created. Both read the same:
 *  who this is, in one glance. The tagline is written for the user; the
 *  charter (identity, boundaries) is written for the expert and stays
 *  under Show more, which opens the full charter in the side panel — or,
 *  where the panel is not available, unfolds it inline. Experts raised
 *  before taglines existed fall back to the charter as their one line. */
export function ExpertChangeCard({
  output,
  artifactId,
  pager,
  decision = null,
  onDecide,
  onUndo,
  action,
  stepDirection,
  waiting = false,
  onSend,
}: Props) {
  const [showCharter, setShowCharter] = useState(false);
  const openArtifact = useCopilotUIStore((s) => s.openArtifact);
  const panelAvailable = !!artifactId;
  const expert = asObject(output.expert) ?? asObject(output.preview);
  if (!expert) return null;

  function handleShowMore() {
    if (panelAvailable && expert && artifactId) {
      openArtifact(toExpertArtifact(expert, output, artifactId));
      return;
    }
    setShowCharter(!showCharter);
  }

  const name = str(expert, "name") ?? "New expert";
  const role = str(expert, "role");
  const tagline = str(expert, "tagline");
  const about = str(expert, "about");
  const boundaries = str(expert, "boundaries");
  const summary = tagline ?? about;
  // The panel always has the full charter to show; inline, the button only
  // earns its place when the one-line summary leaves something out.
  const hasCharter = (!!tagline && !!about) || !!boundaries;
  const showDetails = panelAvailable || hasCharter;
  const showFooter = !!pager || !!onDecide || !!decision || !!action;
  const applied = output.applied === true;
  const header = headerFor(kindOf(output), applied);
  const task = applied ? null : handoffTaskOf(output);
  const outcome = applied ? handoffOutcomeOf(output) : null;
  const stepClassName = cn(
    stepDirection &&
      "duration-300 animate-in fade-in fill-mode-both motion-reduce:animate-none",
    stepDirection === "forward" && "slide-in-from-right-4",
    stepDirection === "back" && "slide-in-from-left-4",
  );

  if (outcome) {
    // Two confirms can share a group (two hires in one send): the pager
    // stays under the status so every expert's outcome can be reached.
    return (
      <div className="flex flex-col gap-2">
        <HandoffStatus
          key={artifactId ?? name}
          outcome={outcome}
          name={name}
          expertId={str(expert, "id")}
          avatarUrl={str(expert, "avatar_url")}
          color={str(expert, "color")}
          onSend={onSend}
        />
        {(!!pager || !!action) && (
          <div className="flex items-center justify-between gap-2">
            <div className="min-w-0">{pager}</div>
            {action}
          </div>
        )}
      </div>
    );
  }

  if (task) {
    return (
      <div className={SHELL}>
        <HandoffProposal
          name={name}
          role={role}
          avatarUrl={str(expert, "avatar_url")}
          color={str(expert, "color")}
          about={summary}
          task={task}
          actions={
            !decision && onDecide
              ? {
                  onStart: () => onDecide("approved"),
                  onNotNow: () => onDecide("declined"),
                  disabled: waiting,
                }
              : null
          }
          onShowProfile={panelAvailable ? handleShowMore : undefined}
          stepClassName={stepClassName}
          stepKey={artifactId ?? name}
        />
        {(!!pager || !!decision || !!action) && (
          <DecisionFooter
            pager={pager}
            decision={decision}
            onUndo={onUndo}
            action={action}
            chosen={
              decision
                ? decision === "approved"
                  ? "Hire and start chosen"
                  : "Not now chosen"
                : undefined
            }
          />
        )}
      </div>
    );
  }

  return (
    <div className={SHELL}>
      <div className={HEADER}>
        <Icon icon={header.icon} size={18} className="text-foreground-subtle" />
        <span className="min-w-0 flex-1 truncate text-sm font-medium text-foreground-strong">
          {header.title}
        </span>
      </div>
      <div className="px-4 py-3">
        <div
          key={artifactId ?? name}
          className={cn(
            "flex items-center gap-2.5",
            stepDirection &&
              "duration-300 animate-in fade-in fill-mode-both motion-reduce:animate-none",
            stepDirection === "forward" && "slide-in-from-right-4",
            stepDirection === "back" && "slide-in-from-left-4",
          )}
        >
          <ExpertAvatar
            name={name}
            avatarUrl={str(expert, "avatar_url")}
            color={str(expert, "color")}
            size={32}
          />
          <div className="min-w-0 flex-1">
            <p className="truncate text-body-sm font-medium leading-inherit text-foreground-emphasis">
              {name}
            </p>
            {role && (
              <p className="truncate text-xs text-foreground-subtle">{role}</p>
            )}
          </div>
          {showDetails && (
            <Button
              variant="secondary"
              size="small"
              aria-expanded={panelAvailable ? undefined : showCharter}
              onClick={handleShowMore}
              leftIcon={<Icon icon={InformationCircleIcon} size={16} />}
              className="h-8 !min-w-0 shrink-0 px-3"
            >
              {showCharter ? "Hide details" : "Details"}
            </Button>
          )}
        </div>
        {summary && (
          <p
            key={`${artifactId ?? name}-summary`}
            className={cn(
              "pl-[42px] pt-2 text-sm text-foreground-tertiary",
              stepDirection &&
                "delay-75 duration-300 animate-in fade-in fill-mode-both motion-reduce:animate-none",
              stepDirection === "forward" && "slide-in-from-right-4",
              stepDirection === "back" && "slide-in-from-left-4",
            )}
          >
            {summary}
          </p>
        )}
        {hasCharter && !panelAvailable && showCharter && (
          <div className="flex flex-col gap-1 pl-[42px] pt-1.5">
            {tagline && about && (
              <p className="text-sm text-foreground-tertiary">{about}</p>
            )}
            {boundaries && (
              <p className="text-sm text-foreground-subtle">
                Stops at: {boundaries}
              </p>
            )}
          </div>
        )}
      </div>
      {showFooter && (
        <DecisionFooter
          pager={pager}
          decision={decision}
          onDecide={onDecide}
          onUndo={onUndo}
          action={action}
          waiting={waiting}
        />
      )}
    </div>
  );
}

const SOUL_LABELS: Record<string, string> = {
  identity: "Identity and personality",
  voice_preferences: "Voice",
  boundaries: "Boundaries",
};

/** A Soul edit the expert proposes for itself, or the one it just saved:
 *  each changed field as it will read, with what it was. Labels match the
 *  team page's Soul editor (SoulDrawer). */
function SoulChangeCard({
  output,
  pager,
  decision = null,
  onDecide,
  onUndo,
  action,
  waiting = false,
}: Props) {
  const showFooter = !!pager || !!onDecide || !!decision || !!action;
  return (
    <div className={SHELL}>
      <div className={HEADER}>
        <Icon
          icon={PencilEdit02Icon}
          size={18}
          className="text-foreground-subtle"
        />
        <span className="min-w-0 flex-1 truncate text-sm font-medium text-foreground-strong">
          {output.applied === true ? "Soul updated" : "Update Soul"}
        </span>
      </div>
      <div className="flex flex-col gap-3 px-4 py-3">
        {soulChanges(output).map((change) => (
          <section key={change.field} className="flex flex-col gap-0.5">
            <p className="text-xs font-medium text-foreground-body">
              {SOUL_LABELS[change.field] ?? change.field}
            </p>
            <p className="whitespace-pre-line text-sm text-foreground-body">
              {change.after || "Empty"}
            </p>
            {change.before && (
              <p className="line-clamp-2 text-xs text-foreground-subtle">
                Was: {change.before}
              </p>
            )}
          </section>
        ))}
      </div>
      {showFooter && (
        <DecisionFooter
          pager={pager}
          decision={decision}
          onDecide={onDecide}
          onUndo={onUndo}
          action={action}
          waiting={waiting}
        />
      )}
    </div>
  );
}

type FooterProps = Pick<
  Props,
  "pager" | "decision" | "onDecide" | "onUndo" | "action" | "waiting"
> & {
  /** What a stored approval will send, for cards offering two approvals. */
  chosen?: string;
};

function DecisionFooter({
  pager,
  decision = null,
  onDecide,
  onUndo,
  action,
  waiting = false,
  chosen,
}: FooterProps) {
  return (
    <div className={FOOTER}>
      <div className="min-w-0">{pager}</div>
      <div className="flex shrink-0 items-center gap-1.5">
        {chosen && (
          <span className="text-xs text-foreground-tertiary">{chosen}</span>
        )}
        {decision && onUndo && (
          <Button
            variant="secondary"
            size="small"
            onClick={onUndo}
            leftIcon={<Icon icon={UndoIcon} size={14} />}
            className="h-8 !min-w-0 px-3"
          >
            {chosen
              ? "Change"
              : decision === "approved"
                ? "Unapprove"
                : "Undo decline"}
          </Button>
        )}
        {!decision && onDecide && (
          <>
            <Button
              variant="ghost"
              size="small"
              disabled={waiting}
              onClick={() => onDecide("declined")}
              className="h-8 !min-w-0 px-3"
            >
              Decline
            </Button>
            <Button
              variant="primary"
              size="small"
              disabled={waiting}
              onClick={() => onDecide("approved")}
              className="h-8 !min-w-0 px-3"
            >
              Approve
            </Button>
          </>
        )}
        {action}
      </div>
    </div>
  );
}

function isSoulChange(output: Record<string, unknown>): boolean {
  return str(output, "type") === "expert_soul_updated";
}

function soulChanges(output: Record<string, unknown>) {
  const changes = Array.isArray(output.changes) ? output.changes : [];
  return changes.flatMap((item) => {
    const change = asObject(item);
    const field = change ? str(change, "field") : null;
    if (!change || !field) return [];
    return [
      {
        field,
        before: str(change, "before") ?? "",
        after: str(change, "after") ?? "",
      },
    ];
  });
}

export function ExpertChangeCardSkeleton() {
  return (
    <div className={SHELL}>
      <div className={HEADER}>
        <Icon
          icon={UserAdd01Icon}
          size={18}
          className="text-foreground-subtle"
        />
        <span className="text-sm font-medium text-foreground-strong">
          Create an Expert
        </span>
      </div>
      <div className="px-4 py-3">
        <div className="flex items-center gap-2.5">
          <Skeleton className="size-8 shrink-0 rounded-full" />
          <div className="flex min-w-0 flex-1 flex-col gap-1.5">
            <Skeleton className="h-3.5 w-28" />
            <Skeleton className="h-3 w-20" />
          </div>
        </div>
        <div className="flex flex-col gap-1.5 pl-[42px] pt-2.5">
          <Skeleton className="h-3 w-full" />
          <Skeleton className="h-3 w-4/5" />
        </div>
      </div>
    </div>
  );
}

interface PartProps {
  part: ToolUIPart;
  isCurrentlyStreaming?: boolean;
  pager?: ReactNode;
  decision?: Decision | null;
  onDecide?: (decision: Decision) => void;
  onUndo?: () => void;
  action?: ReactNode;
  stepDirection?: StepDirection;
  waiting?: boolean;
  onSend?: (message: string) => void;
}

/** The expert is still being written while the row has no output — hold
 *  the card's shape so the real one swaps in without the message jumping.
 *  A row still empty once the stream is over never landed, so it renders
 *  nothing rather than a skeleton that never resolves. */
export function ExpertChangePart({
  part,
  isCurrentlyStreaming,
  pager,
  decision,
  onDecide,
  onUndo,
  action,
  stepDirection,
  waiting,
  onSend,
}: PartProps) {
  if (part.state === "output-error") return <GenericTool part={part} />;
  const output = asObject(part.output);
  if (output) {
    const Card = isSoulChange(output) ? SoulChangeCard : ExpertChangeCard;
    return (
      <Card
        output={output}
        artifactId={part.toolCallId}
        pager={pager}
        decision={decision}
        onDecide={onDecide}
        onUndo={onUndo}
        action={action}
        stepDirection={stepDirection}
        waiting={waiting}
        onSend={onSend}
      />
    );
  }
  // The skeleton is an expert's shape; a Soul edit waits for its diff.
  const soul = SOUL_CHANGE_TOOLS.has(part.type.slice("tool-".length));
  return isCurrentlyStreaming && !soul ? <ExpertChangeCardSkeleton /> : null;
}

function isVisibleExpertPart(
  part: ToolUIPart,
  isCurrentlyStreaming: boolean,
): boolean {
  return (
    part.state === "output-error" ||
    !!asObject(part.output) ||
    isCurrentlyStreaming
  );
}

interface GroupProps {
  parts: ToolUIPart[];
  isCurrentlyStreaming?: boolean;
  /** Share viewer / transcript: the decisions were the owner's to make. */
  readOnly?: boolean;
}

/** One hire/raise per call, so a new team lands as several cards in a
 *  row. They page like the clarifying questions do, one expert at a time:
 *  each Approve/Decline moves to the next, and once every proposal has an
 *  answer a single send posts all of them as one message — the user no
 *  longer reviews it in the composer. A decision can be taken back up to
 *  that point. */
export function ExpertChangeGroup({
  parts,
  isCurrentlyStreaming = false,
  readOnly = false,
}: GroupProps) {
  const [step, setStep] = useState(0);
  const [direction, setDirection] = useState<StepDirection | undefined>();
  const [decisions, setDecisions] = useState<Record<string, Decision>>({});
  const [sent, setSent] = useState(false);
  const actions = useContext(CopilotChatActionsContext);
  const visible = parts.filter((part) =>
    isVisibleExpertPart(part, isCurrentlyStreaming),
  );
  // Otto proposing an expert is the moment to say experts can own tasks.
  // Only live: a decided proposal still reads as open in an old chat.
  useLiveTipMoment("first-expert", {
    happened: !readOnly && visible.some(isOpenExpertProposal),
    isLive: isCurrentlyStreaming,
  });
  if (visible.length === 0) return null;

  const current = Math.min(step, visible.length - 1);
  const part = visible[current];
  const isLast = current === visible.length - 1;
  const proposals = visible
    .map(proposalOf)
    .filter((proposal): proposal is Proposal => proposal !== null);
  const allDecided =
    proposals.length > 0 &&
    proposals.every((proposal) => decisions[proposal.toolCallId]);
  const currentProposal = readOnly ? null : proposalOf(part);
  // While the turn is still streaming another proposal part can land in
  // this group, so the send waits for the stream to settle.
  const sendable = !readOnly && !!actions && !isCurrentlyStreaming;
  // A lone proposal has nothing to review before the send, so its answer
  // is the send: one click hires (and hands off), not Approve then Send.
  const answerSends =
    sendable && visible.length === 1 && proposals.length === 1;

  function goTo(next: number) {
    setDirection(next > current ? "forward" : "back");
    setStep(next);
  }

  function sendMessage(message: string) {
    if (!actions) return;
    // The message is already in the thread for Retry — the failure only
    // needs saying.
    void Promise.resolve(actions.onSend(message)).catch((error: unknown) =>
      toastSendFailure({
        title: "Couldn't send message",
        error,
        recovery: "it is still in the thread, use Retry to send it again",
        paywallRecovery:
          "it is still in the thread, use Retry once you have a plan",
      }),
    );
  }

  function send(answered: Record<string, Decision>) {
    sendMessage(
      proposals
        .map((proposal) => answerLine(proposal, answered[proposal.toolCallId]))
        .join("\n"),
    );
    setSent(true);
  }

  function decide(decision: Decision) {
    if (!currentProposal || isCurrentlyStreaming) return;
    const nextDecisions = {
      ...decisions,
      [currentProposal.toolCallId]: decision,
    };
    setDecisions(nextDecisions);
    if (answerSends) {
      send(nextDecisions);
      return;
    }
    if (!isLast) goTo(current + 1);
  }

  function undo() {
    if (!currentProposal) return;
    setDecisions(withoutDecision(decisions, currentProposal.toolCallId));
  }

  const pager = visible.length > 1 && (
    <div className="flex items-center gap-2">
      <button
        type="button"
        aria-label="Previous expert"
        disabled={current === 0}
        onClick={() => goTo(current - 1)}
        className="flex size-6 items-center justify-center rounded-lg text-foreground-subtle transition-colors enabled:hover:bg-surface-hover enabled:hover:text-foreground-secondary disabled:opacity-35"
      >
        <Icon icon={ArrowLeft01Icon} size={14} />
      </button>
      <span className="flex items-center gap-1.5">
        {visible.map(({ toolCallId }, i) => (
          <button
            key={toolCallId}
            type="button"
            aria-label={`Go to expert ${i + 1}`}
            aria-current={i === current ? "step" : undefined}
            data-decision={decisions[toolCallId]}
            onClick={() => goTo(i)}
            className={cn(
              "rounded-full transition-all duration-300",
              i === current ? "size-2.5 border-2" : "size-2 border",
              decisions[toolCallId] === "approved"
                ? // TODO(tokens): no exact role for border-emerald-500, bg-emerald-500
                  "border-emerald-500 bg-emerald-500"
                : decisions[toolCallId] === "declined"
                  ? // TODO(tokens): no exact role for border-red-400, bg-red-400
                    "border-red-400 bg-red-400"
                  : i === current
                    ? "border-action"
                    : "border-border-strong",
            )}
          />
        ))}
      </span>
      <button
        type="button"
        aria-label="Next expert"
        disabled={isLast}
        onClick={() => goTo(current + 1)}
        className="flex size-6 items-center justify-center rounded-lg text-foreground-subtle transition-colors enabled:hover:bg-surface-hover enabled:hover:text-foreground-secondary disabled:opacity-35"
      >
        <Icon icon={ArrowRight01Icon} size={14} />
      </button>
      <span className="text-xs text-foreground-subtle">
        {current + 1} of {visible.length}
      </span>
    </div>
  );

  const canSend = sendable && allDecided;
  const action = sent ? (
    <span className="flex items-center gap-1 text-xs text-foreground-subtle">
      <Icon icon={Tick02Icon} size={14} />
      Sent
    </span>
  ) : canSend ? (
    <Button
      variant="primary"
      size="icon"
      aria-label="Send decisions"
      onClick={() => send(decisions)}
      className="size-8 p-0"
    >
      <Icon icon={SentIcon} size={15} />
    </Button>
  ) : null;

  // A lone hand-off card, once answered, is only the record of the choice:
  // the expert's own card below says where the work went.
  const lone = proposals.length === 1 && visible.length === 1;
  if (sent && lone && proposals[0].handoff) {
    return (
      <div className="my-2">
        <HandoffReceipt>
          {receiptFor(proposals[0], decisions[proposals[0].toolCallId])}
        </HandoffReceipt>
      </div>
    );
  }

  // The shell (Details, pager, decision buttons) is the same from one
  // expert to the next, so it stays mounted; only the identity block
  // inside the card slides — see ExpertChangeCard.
  return (
    <div className="my-2">
      <ExpertChangePart
        part={part}
        isCurrentlyStreaming={isCurrentlyStreaming}
        pager={pager || undefined}
        decision={decisions[part.toolCallId] ?? null}
        onDecide={currentProposal && !sent ? decide : undefined}
        onUndo={currentProposal && !sent ? undo : undefined}
        action={action || undefined}
        stepDirection={direction}
        waiting={isCurrentlyStreaming}
        onSend={sendable ? sendMessage : undefined}
      />
    </div>
  );
}

/** The record of the choice, worded as the choice: the confirm runs after
 *  the send and can still fail, and the expert's own card says what came of
 *  it. Only a hire proposed for a task shrinks to a receipt. */
function receiptFor(proposal: Proposal, decision: Decision | undefined) {
  if (decision === "approved" && proposal.handoff) {
    return `You chose to hire ${proposal.name} to take on ${proposal.handoff.title}.`;
  }
  return `You chose not to hire ${proposal.name} for now.`;
}
