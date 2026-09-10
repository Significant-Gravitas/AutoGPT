"use client";

import { useGetExpert } from "@/app/api/__generated__/endpoints/experts/experts";
import { okData } from "@/app/api/helpers";
import { swatchClassFor } from "@/app/(platform)/raise/components/ColorStep/helpers";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import { cn } from "@/lib/utils";
import type { ExpertArtifact } from "../../../store";

interface Props {
  expert: ExpertArtifact;
}

interface SectionProps {
  title: string;
  text: string;
}

// Same section labels as the team page's soul editor (SoulDrawer), so the
// charter reads the same here as where it gets edited.
const LABEL =
  "mb-1.5 text-xs font-medium uppercase tracking-[0.12em] text-zinc-700";

function Section({ title, text }: SectionProps) {
  return (
    <section>
      <div className={LABEL}>{title}</div>
      <p className="whitespace-pre-line text-sm leading-relaxed text-zinc-600">
        {text}
      </p>
    </section>
  );
}

const KIND_LABELS: Record<NonNullable<ExpertArtifact["kind"]>, string> = {
  hire: "Hired from the roster",
  raise: "Raised from a charter",
  update: "Charter update",
};

/** Messages saved before the applied response carried the charter only
 *  hold the summary — for those, the expert record itself fills the gaps. */
function useCharterFallback(artifact: ExpertArtifact): ExpertArtifact {
  const missing = !!artifact.id && !artifact.about && !artifact.boundaries;
  const query = useGetExpert(artifact.id ?? "", {
    query: { select: (res) => okData(res) ?? null, enabled: missing },
  });
  const record = missing ? query.data : null;
  if (!record) return artifact;
  return {
    ...artifact,
    role: artifact.role ?? record.role,
    tagline: artifact.tagline ?? record.tagline ?? null,
    about: record.identity || null,
    boundaries: record.boundaries || null,
    voicePreferences:
      artifact.voicePreferences ?? (record.voice_preferences || null),
    weeklyBudget: artifact.weeklyBudget ?? record.weekly_budget ?? null,
    avatarUrl: artifact.avatarUrl ?? record.avatar_url ?? null,
    color: artifact.color ?? (record.color || null),
  };
}

/** The whole charter of a proposed or newly created expert — the card in
 *  the chat shows one line, this is everything the model wrote for them. */
export function ExpertArtifactContent({ expert: artifact }: Props) {
  const expert = useCharterFallback(artifact);
  const status = expert.applied
    ? "On the team"
    : "Proposed — nothing created yet";

  return (
    <div className="flex flex-1 flex-col gap-6 overflow-y-auto p-5">
      <div className="flex items-center gap-3">
        <ExpertAvatar
          name={expert.name}
          avatarUrl={expert.avatarUrl}
          size={48}
        />
        <div className="min-w-0">
          <p className="truncate text-base font-semibold text-zinc-900">
            {expert.name}
          </p>
          {expert.role && (
            <p className="truncate text-sm text-zinc-500">{expert.role}</p>
          )}
        </div>
      </div>
      {expert.tagline && (
        <p className="text-sm leading-relaxed text-zinc-700">
          {expert.tagline}
        </p>
      )}
      <Section title="Status" text={status} />
      {expert.kind && <Section title="Kind" text={KIND_LABELS[expert.kind]} />}
      {expert.about && (
        <Section title="Identity and personality" text={expert.about} />
      )}
      {expert.voicePreferences && (
        <Section title="Voice" text={expert.voicePreferences} />
      )}
      {expert.boundaries && (
        <Section title="Boundaries" text={expert.boundaries} />
      )}
      <Section
        title="Weekly budget"
        text={
          expert.weeklyBudget === null
            ? "Default"
            : `${expert.weeklyBudget} credits`
        }
      />
      {expert.color && swatchClassFor(expert.color) && (
        <section>
          <div className={LABEL}>Accent</div>
          <div className="flex items-center gap-2 text-sm text-zinc-600">
            <span
              aria-hidden="true"
              className={cn(
                "size-4 rounded-full",
                swatchClassFor(expert.color),
              )}
            />
            {expert.color}
          </div>
        </section>
      )}
      {expert.id && <Section title="Expert ID" text={expert.id} />}
    </div>
  );
}
