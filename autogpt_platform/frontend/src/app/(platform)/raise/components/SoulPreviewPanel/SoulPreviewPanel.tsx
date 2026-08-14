"use client";

import { Avatar, AvatarFallback } from "@/components/atoms/Avatar/Avatar";
import { raisedIdentity } from "../../helpers";
import { expertInitials } from "./helpers";

interface Props {
  name: string;
  voiceLabel: string | null;
  firstJobName: string | null;
}

export function SoulPreviewPanel({ name, voiceLabel, firstJobName }: Props) {
  return (
    <aside className="flex flex-col overflow-hidden rounded-3xl border border-zinc-200 bg-white shadow-sm">
      <div className="flex items-center gap-3 border-b border-zinc-200 px-6 py-5">
        <Avatar className="h-11 w-11">
          <AvatarFallback>{expertInitials(name)}</AvatarFallback>
        </Avatar>
        <div>
          <h2 className="text-lg font-semibold tracking-[-0.01em] text-zinc-900">
            {name ? `${name}'s Soul` : "A new Soul"}
          </h2>
          <p className="text-sm text-zinc-500">Edit any line, anytime.</p>
        </div>
      </div>

      <dl className="flex flex-col gap-5 px-6 py-6">
        <SoulLine
          label="Name"
          value={name || null}
          placeholder="Waiting for a name…"
        />
        <SoulLine
          label="Identity"
          value={name ? raisedIdentity(name) : null}
          placeholder="Takes shape once you name me."
        />
        <SoulLine
          label="Voice"
          value={voiceLabel}
          placeholder="However you'd like me to sound."
        />
        <SoulLine
          label="First job"
          value={firstJobName}
          placeholder="Something to get started on."
        />
      </dl>
    </aside>
  );
}

interface SoulLineProps {
  label: string;
  value: string | null;
  placeholder: string;
}

function SoulLine({ label, value, placeholder }: SoulLineProps) {
  return (
    <div className="flex flex-col gap-1">
      <dt className="text-xs font-medium uppercase tracking-[0.12em] text-purple-600">
        {label}
      </dt>
      <dd
        className={
          value
            ? "text-[15px] text-zinc-800"
            : "text-[15px] italic text-zinc-400"
        }
      >
        {value ?? placeholder}
      </dd>
    </div>
  );
}
