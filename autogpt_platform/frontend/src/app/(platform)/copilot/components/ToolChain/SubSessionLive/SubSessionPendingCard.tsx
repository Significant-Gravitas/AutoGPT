import { Icon } from "@/components/atoms/Icon/Icon";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import { cn } from "@/lib/utils";
import { LinkSquare01Icon } from "@hugeicons/core-free-icons";
import Link from "next/link";
import { useExpertMap } from "../../../useExpertMap";
import { CARD, StatusPill } from "../ResultCards";
import { asObject, str } from "../resultHelpers";
import { SubSessionLive } from "./SubSessionLive";
import { useDelegatedSessionId } from "./useDelegatedSessionId";
import { useLiveSubSession } from "./useLiveSubSession";

/** The card for a delegated run that hasn't returned yet. A blocking
 *  delegate/handoff yields no tool output while the teammate works, so
 *  everything comes from the tool INPUT: delegate/handoff carry expert_id +
 *  prompt, get_sub_session_result carries the exact sub_session_id. When
 *  only the expert is known, their currently-processing session is found by
 *  polling their session list; when only the session is known, the expert
 *  is read off the polled session itself. Swapped for the full
 *  SubSessionCard the moment the tool returns. */
interface PendingCardProps {
  input: unknown;
  minimal?: boolean;
}

export function SubSessionPendingCard({
  input,
  minimal = false,
}: PendingCardProps) {
  const { expertsById } = useExpertMap();
  const args = asObject(input) ?? {};
  const inputExpertId = str(args, "expert_id");
  const prompt = str(args, "prompt");
  const inputSessionId = str(args, "sub_session_id") ?? str(args, "session_id");
  const discoveredId = useDelegatedSessionId(
    inputSessionId ? null : inputExpertId,
  );
  const liveSessionId = inputSessionId ?? discoveredId;
  // Same query key as the live view below, so a full card polls once.
  const {
    session: liveSession,
    isError,
    isPaused,
  } = useLiveSubSession(liveSessionId ?? "", !!liveSessionId);
  const expertId = inputExpertId ?? liveSession?.expert_id ?? null;
  const expert = expertId ? expertsById.get(expertId) : undefined;

  return (
    <div className={cn(CARD, "w-full rounded-2xl p-2.5")}>
      <div className="flex items-center gap-2.5">
        <ExpertAvatar
          name={expert?.name ?? "Sub-AutoPilot"}
          avatarUrl={expert?.avatarUrl ?? null}
          size={28}
        />
        <p className="min-w-0 flex-1 truncate text-sm font-medium text-zinc-800">
          {expert?.name ?? "Sub-AutoPilot"}
          {expert?.role && (
            <span className="ml-1.5 font-normal text-zinc-400">
              {expert.role}
            </span>
          )}
        </p>
        {/* Once the poll dies the card no longer knows the run is going —
            keeping the spinner up would be a guess dressed as a fact. */}
        <StatusPill
          status={isError || isPaused ? "unknown" : "running"}
          className="text-sm"
        />
        {liveSessionId && (
          <Link
            href={`/copilot?sessionId=${liveSessionId}`}
            aria-label="Open sub-session"
            className="shrink-0 rounded-full p-1 text-zinc-400 transition-colors hover:bg-zinc-100 hover:text-zinc-700"
          >
            <Icon icon={LinkSquare01Icon} size={14} />
          </Link>
        )}
      </div>
      {!minimal && prompt && (
        <p className="mt-1.5 line-clamp-2 pl-9 text-sm text-zinc-500">
          {prompt}
        </p>
      )}
      {!minimal && liveSessionId && (
        <SubSessionLive subSessionId={liveSessionId} active />
      )}
    </div>
  );
}
