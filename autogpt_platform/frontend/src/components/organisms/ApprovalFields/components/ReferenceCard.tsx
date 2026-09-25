import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import { factText, type Reference } from "../helpers";

// Named, then a count: the marketplace expert card's skills, in a line.
const NAMED_SKILLS = 3;

interface Props {
  reference: Reference;
}

export function ReferenceCard({ reference }: Props) {
  const facts = reference.kind
    ? reference.meta.map(factText).join(" · ")
    : reference.summary;
  const hasAvatar = Boolean(reference.avatarURL || reference.avatarColor);
  return (
    <div>
      <div className="flex items-center gap-2.5">
        {hasAvatar && (
          <ExpertAvatar
            name={reference.name}
            avatarUrl={reference.avatarURL}
            color={reference.avatarColor}
            size={36}
            className="shrink-0 rounded-full"
          />
        )}
        <div className="min-w-0">
          {reference.kind && <p className="text-zinc-500">{reference.kind}</p>}
          <p translate="no" className="text-sm font-semibold text-zinc-900">
            {reference.name}
          </p>
        </div>
      </div>
      {reference.description && (
        <p className="mt-1 line-clamp-3 text-zinc-700">
          {reference.description}
        </p>
      )}
      {facts && <p className="mt-1.5 text-zinc-500">{facts}</p>}
      {reference.skills.length > 0 && (
        <p className="mt-1.5 text-zinc-700">
          <span className="text-zinc-500">Skills: </span>
          {skillsText(reference.skills)}
        </p>
      )}
      <p
        translate="no"
        className="mt-1.5 font-mono text-[0.6875rem] text-zinc-400"
      >
        {reference.id}
      </p>
    </div>
  );
}

function skillsText(skills: string[]) {
  const named = skills.slice(0, NAMED_SKILLS).join(", ");
  const rest = skills.length - NAMED_SKILLS;
  return rest > 0 ? `${named} +${rest} more` : named;
}
