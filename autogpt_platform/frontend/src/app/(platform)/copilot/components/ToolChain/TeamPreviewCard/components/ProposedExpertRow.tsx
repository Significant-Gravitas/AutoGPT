"use client";

import {
  ArrowDown01Icon,
  ArrowTurnBackwardIcon,
  Cancel01Icon,
  CheckmarkCircle02Icon,
} from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import { ACCORDION_PANEL, accordionState } from "../../accordion";
import type { TeamProposal } from "../../helpers";
import { str } from "../../resultHelpers";
import { proposalName } from "../helpers";

interface Props {
  proposal: TeamProposal;
  confirmed: boolean;
  removed: boolean;
  open: boolean;
  founderMode?: boolean;
  onToggleRemoved: () => void;
  onToggleOpen: () => void;
}

export function ProposedExpertRow({
  proposal,
  confirmed,
  removed,
  open,
  founderMode = false,
  onToggleRemoved,
  onToggleOpen,
}: Props) {
  const { preview } = proposal;
  const name = proposalName(proposal);
  const role = str(preview, "role");
  const about = str(preview, "about");
  const boundaries = str(preview, "boundaries");
  const voice = str(preview, "voice_preferences");
  const budget =
    typeof preview.weekly_budget === "number" ? preview.weekly_budget : null;
  const hasCharter =
    !!(about || boundaries || voice) || (!founderMode && budget !== null);

  return (
    <div className={"py-2.5 " + (removed ? "opacity-40" : "")}>
      <div className="flex items-center gap-2.5">
        <ExpertAvatar
          name={name}
          avatarUrl={str(preview, "avatar_url")}
          size={28}
        />
        <button
          type="button"
          onClick={onToggleOpen}
          disabled={!hasCharter}
          aria-expanded={open}
          className="flex min-w-0 flex-1 items-center gap-1.5 text-left"
        >
          <span
            className={
              "shrink-0 text-[13px] font-medium text-zinc-800 " +
              (removed ? "line-through" : "")
            }
          >
            {name}
          </span>
          {role && (
            <span className="min-w-0 truncate text-xs text-zinc-400">
              {role}
            </span>
          )}
          {hasCharter && (
            <Icon
              icon={ArrowDown01Icon}
              size={12}
              className={
                "shrink-0 text-zinc-400 transition-transform duration-300 ease-out-quint " +
                (open ? "rotate-180" : "")
              }
            />
          )}
        </button>
        {confirmed ? (
          <span className="inline-flex shrink-0 items-center gap-1 rounded-md bg-emerald-50 px-1.5 py-0.5 text-[10px] font-semibold text-emerald-700">
            <Icon icon={CheckmarkCircle02Icon} size={11} />
            Hired
          </span>
        ) : (
          <button
            type="button"
            onClick={onToggleRemoved}
            aria-label={removed ? `Put ${name} back` : `Remove ${name}`}
            className="flex size-7 shrink-0 items-center justify-center rounded-full text-zinc-400 transition-colors hover:bg-zinc-100 hover:text-zinc-700"
          >
            <Icon
              icon={removed ? ArrowTurnBackwardIcon : Cancel01Icon}
              size={14}
            />
          </button>
        )}
      </div>
      {hasCharter && (
        <div className={ACCORDION_PANEL + " " + accordionState(open)}>
          <div aria-hidden={!open} className="min-h-0 overflow-hidden">
            <div className="flex flex-col gap-1 pl-[38px] pt-1.5 text-xs text-zinc-500">
              {about && <p>{about}</p>}
              {boundaries && (
                <p className="text-zinc-400">Stops at: {boundaries}</p>
              )}
              {voice && <p className="text-zinc-400">Sounds like: {voice}</p>}
              {!founderMode && budget !== null && (
                <p className="text-zinc-400">Weekly budget: {budget} credits</p>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
