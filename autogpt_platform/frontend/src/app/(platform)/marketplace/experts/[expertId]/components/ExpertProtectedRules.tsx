import { Icon } from "@/components/atoms/Icon/Icon";
import { CheckmarkCircle02Icon, LockIcon } from "@hugeicons/core-free-icons";
import { ExpertNoteCard } from "./ExpertNoteCard";
import { ExpertSection } from "./ExpertSection";

interface Props {
  rules: string[];
}

/** The rules baked into every expert's soul, quoted verbatim — the same
 *  strings the backend injects, so the promise on the profile and the
 *  instruction the model receives can never drift apart. */
export function ExpertProtectedRules({ rules }: Props) {
  if (rules.length === 0) return null;

  return (
    <ExpertSection
      title="Rules this expert cannot break"
      description="Part of every expert's soul, and not editable by anyone."
    >
      <ExpertNoteCard icon={LockIcon}>
        <ul className="flex flex-col gap-2">
          {rules.map((rule) => (
            <li
              key={rule}
              className="flex gap-2.5 text-base leading-7 text-zinc-600"
            >
              <Icon
                icon={CheckmarkCircle02Icon}
                size={18}
                aria-hidden="true"
                className="mt-1 shrink-0 text-emerald-600"
              />
              <span>{rule}</span>
            </li>
          ))}
        </ul>
      </ExpertNoteCard>
    </ExpertSection>
  );
}
