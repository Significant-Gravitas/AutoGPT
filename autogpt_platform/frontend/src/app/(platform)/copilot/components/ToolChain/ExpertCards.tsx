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
import { type ReactNode, useState } from "react";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import { cn } from "@/lib/utils";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { GenericTool } from "../../tools/GenericTool/GenericTool";
import { type ArtifactRef, useCopilotUIStore } from "../../store";
import { asObject, str } from "./resultHelpers";

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
  /** Footer control the group adds once every proposal is decided. */
  action?: ReactNode;
  /** Which way the pager just moved — the identity block slides in from
   *  that side while the shell (Details, pager, decision) stays put. */
  stepDirection?: StepDirection;
}

export type StepDirection = "forward" | "back";

export type Decision = "approved" | "declined";

// Same shell as the chain's action card (connect / inputs / questions) so
// every "your turn" card in the thread reads as one family.
const SHELL =
  "w-full overflow-hidden rounded-3xl border border-zinc-100 bg-white";
const HEADER = "flex items-center gap-2.5 border-b border-zinc-100 px-4 py-3";
const FOOTER =
  "flex items-center justify-between gap-2 border-t border-zinc-100 px-4 py-3";

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
            : "Expert raised",
    };
  }
  return {
    icon,
    title:
      kind === "hire"
        ? "Hire an expert"
        : kind === "update"
          ? "Update an expert"
          : "Raise an expert",
  };
}

interface Proposal {
  toolCallId: string;
  name: string;
  confirmationId: string;
}

/** A hire/raise preview the user can still say yes or no to. */
function proposalOf(part: ToolUIPart): Proposal | null {
  const output = asObject(part.output);
  if (!output || output.applied === true) return null;
  const confirmationId = str(output, "confirmation_id");
  const preview = asObject(output.preview);
  if (!confirmationId || !preview) return null;
  return {
    toolCallId: part.toolCallId,
    name: str(preview, "name") ?? "this expert",
    confirmationId,
  };
}

function decisionLine(proposal: Proposal, decision: Decision): string {
  return decision === "approved"
    ? `Approved: create ${proposal.name} (confirmation_id: ${proposal.confirmationId}).`
    : `Not approved: do not create ${proposal.name}, discard that proposal (confirmation_id: ${proposal.confirmationId}).`;
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
}: Props) {
  const [showCharter, setShowCharter] = useState(false);
  const openArtifact = useCopilotUIStore((s) => s.openArtifact);
  const panelAvailable = useGetFlag(Flag.ARTIFACTS) && !!artifactId;
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
  const hasCharter =
    (!!tagline && !!about) ||
    !!boundaries ||
    !!str(expert, "voice_preferences") ||
    typeof expert.weekly_budget === "number";
  const showDetails = panelAvailable || hasCharter;
  const showFooter = !!pager || !!onDecide || !!decision || !!action;
  const header = headerFor(kindOf(output), output.applied === true);

  return (
    <div className={SHELL}>
      <div className={HEADER}>
        <Icon icon={header.icon} size={18} className="text-zinc-400" />
        <span className="min-w-0 flex-1 truncate text-sm font-medium text-zinc-900">
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
            size={32}
          />
          <div className="min-w-0 flex-1">
            <p className="truncate text-[13px] font-medium text-zinc-800">
              {name}
            </p>
            {role && <p className="truncate text-xs text-zinc-400">{role}</p>}
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
              "pl-[42px] pt-2 text-sm text-zinc-500",
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
              <p className="text-sm text-zinc-500">{about}</p>
            )}
            {boundaries && (
              <p className="text-sm text-zinc-400">Stops at: {boundaries}</p>
            )}
          </div>
        )}
      </div>
      {showFooter && (
        <div className={FOOTER}>
          <div className="min-w-0">{pager}</div>
          <div className="flex shrink-0 items-center gap-1.5">
            {decision && onUndo && (
              <Button
                variant="secondary"
                size="small"
                onClick={onUndo}
                leftIcon={<Icon icon={UndoIcon} size={14} />}
                className="h-8 !min-w-0 px-3"
              >
                {decision === "approved" ? "Unapprove" : "Undo decline"}
              </Button>
            )}
            {!decision && onDecide && (
              <>
                <Button
                  variant="ghost"
                  size="small"
                  onClick={() => onDecide("declined")}
                  className="h-8 !min-w-0 px-3"
                >
                  Decline
                </Button>
                <Button
                  variant="primary"
                  size="small"
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
      )}
    </div>
  );
}

export function ExpertChangeCardSkeleton() {
  return (
    <div className={SHELL}>
      <div className={HEADER}>
        <Icon icon={UserAdd01Icon} size={18} className="text-zinc-400" />
        <span className="text-sm font-medium text-zinc-900">
          Raise an expert
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
}: PartProps) {
  if (part.state === "output-error") return <GenericTool part={part} />;
  const output = asObject(part.output);
  if (output) {
    return (
      <ExpertChangeCard
        output={output}
        artifactId={part.toolCallId}
        pager={pager}
        decision={decision}
        onDecide={onDecide}
        onUndo={onUndo}
        action={action}
        stepDirection={stepDirection}
      />
    );
  }
  return isCurrentlyStreaming ? <ExpertChangeCardSkeleton /> : null;
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
 *  answer a single send drafts all of them into the composer — the user
 *  still reads and sends the message themselves. */
export function ExpertChangeGroup({
  parts,
  isCurrentlyStreaming = false,
  readOnly = false,
}: GroupProps) {
  const [step, setStep] = useState(0);
  const [direction, setDirection] = useState<StepDirection | undefined>();
  const [decisions, setDecisions] = useState<Record<string, Decision>>({});
  const [drafted, setDrafted] = useState(false);
  const setInitialPrompt = useCopilotUIStore((s) => s.setInitialPrompt);
  const visible = parts.filter((part) =>
    isVisibleExpertPart(part, isCurrentlyStreaming),
  );
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

  function goTo(next: number) {
    setDirection(next > current ? "forward" : "back");
    setStep(next);
  }

  function decide(decision: Decision) {
    if (!currentProposal) return;
    setDecisions({ ...decisions, [currentProposal.toolCallId]: decision });
    if (!isLast) goTo(current + 1);
  }

  function undo() {
    if (!currentProposal) return;
    const { [currentProposal.toolCallId]: _, ...rest } = decisions;
    setDecisions(rest);
    // The draft in the composer no longer matches — offer the send again
    // once every proposal has an answer.
    setDrafted(false);
  }

  function draft() {
    setInitialPrompt(
      proposals
        .map((proposal) =>
          decisionLine(proposal, decisions[proposal.toolCallId]),
        )
        .join("\n"),
    );
    setDrafted(true);
  }

  const pager = visible.length > 1 && (
    <div className="flex items-center gap-2">
      <button
        type="button"
        aria-label="Previous expert"
        disabled={current === 0}
        onClick={() => goTo(current - 1)}
        className="flex size-6 items-center justify-center rounded-lg text-zinc-400 transition-colors enabled:hover:bg-zinc-100 enabled:hover:text-zinc-600 disabled:opacity-35"
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
                ? "border-emerald-500 bg-emerald-500"
                : decisions[toolCallId] === "declined"
                  ? "border-red-400 bg-red-400"
                  : i === current
                    ? "border-zinc-800"
                    : "border-zinc-300",
            )}
          />
        ))}
      </span>
      <button
        type="button"
        aria-label="Next expert"
        disabled={isLast}
        onClick={() => goTo(current + 1)}
        className="flex size-6 items-center justify-center rounded-lg text-zinc-400 transition-colors enabled:hover:bg-zinc-100 enabled:hover:text-zinc-600 disabled:opacity-35"
      >
        <Icon icon={ArrowRight01Icon} size={14} />
      </button>
      <span className="text-xs text-zinc-400">
        {current + 1} of {visible.length}
      </span>
    </div>
  );

  const action =
    !readOnly &&
    allDecided &&
    (drafted ? (
      <span className="flex items-center gap-1 text-xs text-zinc-400">
        <Icon icon={Tick02Icon} size={14} />
        Added to message
      </span>
    ) : (
      <Button
        variant="primary"
        size="icon"
        aria-label="Add decisions to message"
        onClick={draft}
        className="size-8 p-0"
      >
        <Icon icon={SentIcon} size={15} />
      </Button>
    ));

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
        onDecide={currentProposal ? decide : undefined}
        onUndo={currentProposal ? undo : undefined}
        action={action || undefined}
        stepDirection={direction}
      />
    </div>
  );
}
