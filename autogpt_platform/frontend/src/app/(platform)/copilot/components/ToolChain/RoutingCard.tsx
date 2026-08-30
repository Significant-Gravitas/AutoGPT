"use client";

import Link from "next/link";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import { useCopilotUIStore } from "@/app/(platform)/copilot/store";
import { cn } from "@/lib/utils";
import { CARD, StatusPill } from "./ResultCards";
import { asObject, str } from "./resultHelpers";
import { useSubSessionEffectiveStatus } from "./SubSessionLive";

interface Props {
  output: Record<string, unknown>;
}

/** A delegation that actually went out: who has it, which task receipt it
 *  opened, and the door into that receipt's drawer on the Team page. */
export function RoutingCard({ output }: Props) {
  const expert = asObject(output.expert);
  const taskId = str(output, "task_id");
  const taskTitle = str(output, "task_title");
  const subSessionId = str(output, "sub_session_id");
  const status = useSubSessionEffectiveStatus(
    subSessionId,
    str(output, "status"),
  );
  if (!expert || !taskId) return null;

  const name = str(expert, "name") ?? "Teammate";
  const role = str(expert, "role");

  return (
    <div className={cn(CARD, "w-full rounded-2xl p-2.5")}>
      <div className="flex items-center gap-2.5">
        <ExpertAvatar
          name={name}
          avatarUrl={str(expert, "avatar_url") ?? null}
          size={28}
        />
        <div className="min-w-0 flex-1">
          <p className="truncate text-[13px] font-medium text-zinc-800">
            Sent to {name}
            {role && (
              <span className="ml-1.5 font-normal text-zinc-400">{role}</span>
            )}
          </p>
          <p className="truncate text-xs text-zinc-500">
            {taskTitle ?? "Delegated task"}
          </p>
        </div>
        {status && <StatusPill status={status} />}
        <Link
          href={`/team?task=${encodeURIComponent(taskId)}`}
          className="shrink-0 text-xs text-zinc-500 transition-colors hover:text-zinc-800"
        >
          View task
        </Link>
      </div>
    </div>
  );
}

/** The router's uncertain path: a proposed delegation nothing was sent for
 *  yet. Accept / pick-someone-else draft the user's reply into the composer
 *  (never auto-send), and the accepted delegation arrives as a fresh
 *  ``delegate_to_expert`` call without the confirmation flag. */
export function RoutingConfirmCard({ output }: Props) {
  const setInitialPrompt = useCopilotUIStore((s) => s.setInitialPrompt);
  const expert = asObject(output.expert);
  if (!expert) return null;

  const name = str(expert, "name") ?? "Teammate";
  const role = str(expert, "role");
  const taskTitle = str(output, "task_title") ?? "this task";

  return (
    <div className={cn(CARD, "w-full rounded-2xl p-2.5")}>
      <div className="flex items-center gap-2.5">
        <ExpertAvatar
          name={name}
          avatarUrl={str(expert, "avatar_url") ?? null}
          size={28}
        />
        <div className="min-w-0 flex-1">
          <p className="truncate text-[13px] font-medium text-zinc-800">
            Send to {name}?
            {role && (
              <span className="ml-1.5 font-normal text-zinc-400">{role}</span>
            )}
          </p>
          <p className="truncate text-xs text-zinc-500">{taskTitle}</p>
        </div>
      </div>
      <div className="mt-2 flex items-center gap-2 pl-9">
        <button
          type="button"
          onClick={() => setInitialPrompt(`Yes — send it to ${name}.`)}
          className="rounded-full bg-zinc-900 px-3 py-1 text-xs font-medium text-white transition-colors hover:bg-zinc-700"
        >
          Accept
        </button>
        <button
          type="button"
          onClick={() =>
            setInitialPrompt(`Not ${name} — let's pick someone else for this.`)
          }
          className="rounded-full px-3 py-1 text-xs font-medium text-zinc-600 ring-1 ring-inset ring-zinc-200 transition-colors hover:bg-zinc-50"
        >
          Pick someone else
        </button>
      </div>
    </div>
  );
}
