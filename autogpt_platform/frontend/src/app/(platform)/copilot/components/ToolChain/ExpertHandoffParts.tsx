"use client";

import { ArrowDown01Icon, Tick02Icon } from "@hugeicons/core-free-icons";
import { type ReactNode, useId, useState } from "react";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import { cn } from "@/lib/utils";
import type { HandoffTask } from "./expertHandoff";

interface ProposalProps {
  name: string;
  role: string | null;
  avatarUrl: string | null;
  color: string | null;
  /** One line about the expert, written for the user. */
  about: string | null;
  task: HandoffTask;
  /** Absent once answered, or on a read-only transcript. */
  actions: {
    onStart: () => void;
    onNotNow: () => void;
    disabled: boolean;
  } | null;
  /** Opens the full charter in the side panel, where there is one. */
  onShowProfile?: () => void;
  /** Classes that slide the identity in when a pager moves. */
  stepClassName?: string;
  stepKey: string;
}

/** A hire proposed for a task the user already described, asked as one
 *  yes/no question: Hire and start, or Not now. The brief and the expert's
 *  charter stay behind one disclosure: the user wrote the task a moment ago
 *  and only needs who, what and the choice to answer. */
export function HandoffProposal({
  name,
  role,
  avatarUrl,
  color,
  about,
  task,
  actions,
  onShowProfile,
  stepClassName,
  stepKey,
}: ProposalProps) {
  return (
    <div className="flex gap-3 p-4">
      <ExpertAvatar
        key={`${stepKey}-avatar`}
        name={name}
        avatarUrl={avatarUrl}
        color={color}
        size={40}
      />
      <div className="min-w-0 flex-1">
        <div key={stepKey} className={stepClassName}>
          <p className="text-body font-medium leading-snug text-foreground-emphasis [overflow-wrap:anywhere]">
            Hire {name} to take on {task.title}?
          </p>
          <p className="mt-0.5 text-sm text-foreground-tertiary [overflow-wrap:anywhere]">
            {role
              ? `${role} · works on it in their own chat`
              : "Works on it in their own chat"}
          </p>
        </div>
        {actions && (
          <div className="mt-3 flex flex-wrap items-center gap-2">
            <Button
              variant="primary"
              size="small"
              disabled={actions.disabled}
              onClick={actions.onStart}
              className="h-8 !min-w-0 px-3.5"
            >
              Hire and start
            </Button>
            <Button
              variant="ghost"
              size="small"
              disabled={actions.disabled}
              onClick={actions.onNotNow}
              className="h-8 !min-w-0 px-3"
            >
              Not now
            </Button>
          </div>
        )}
        <HandoffDetails
          name={name}
          about={about}
          task={task}
          onShowProfile={onShowProfile}
        />
      </div>
    </div>
  );
}

interface DetailsProps {
  name: string;
  about: string | null;
  task: HandoffTask;
  onShowProfile?: () => void;
}

function HandoffDetails({ name, about, task, onShowProfile }: DetailsProps) {
  const [open, setOpen] = useState(false);
  const panelId = useId();
  return (
    <div className="mt-3">
      <button
        type="button"
        aria-expanded={open}
        aria-controls={panelId}
        onClick={() => setOpen(!open)}
        className="inline-flex items-center gap-1 rounded text-xs font-medium text-foreground-secondary underline-offset-2 hover:underline focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-focus"
      >
        What {name} will get
        <Icon
          icon={ArrowDown01Icon}
          size={12}
          aria-hidden
          className={cn(
            "transition-transform motion-reduce:transition-none",
            open && "rotate-180",
          )}
        />
      </button>
      <div
        id={panelId}
        hidden={!open}
        className={cn("mt-2 flex-col gap-2", open ? "flex" : "hidden")}
      >
        <p className="whitespace-pre-line rounded-2xl bg-surface-muted px-3 py-2.5 text-sm text-foreground-tertiary [overflow-wrap:anywhere]">
          {task.brief}
        </p>
        {about && (
          <p className="text-xs text-foreground-subtle [overflow-wrap:anywhere]">
            About {name}: {about}
          </p>
        )}
        {onShowProfile && (
          <button
            type="button"
            onClick={onShowProfile}
            className="self-start rounded text-xs font-medium text-foreground-secondary underline-offset-2 hover:underline focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-focus"
          >
            See {name}&apos;s full profile
          </button>
        )}
      </div>
    </div>
  );
}

interface ReceiptProps {
  children: ReactNode;
}

/** What an answered hand-off card shrinks to: the record of the choice. */
export function HandoffReceipt({ children }: ReceiptProps) {
  return (
    <p className="flex items-center gap-2 text-sm text-foreground-tertiary">
      <span className="flex size-5 shrink-0 items-center justify-center rounded-full bg-surface-muted">
        <Icon icon={Tick02Icon} size={12} aria-hidden />
      </span>
      <span className="min-w-0 [overflow-wrap:anywhere]">{children}</span>
    </p>
  );
}
