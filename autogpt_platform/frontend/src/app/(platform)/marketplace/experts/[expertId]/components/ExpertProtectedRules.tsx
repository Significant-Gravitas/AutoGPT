import { Icon } from "@/components/atoms/Icon/Icon";
import { LockIcon, SecurityCheckIcon } from "@hugeicons/core-free-icons";
import { ExpertSection } from "./ExpertSection";

interface Props {
  rules: string[];
}

/** The rules baked into every expert's soul, quoted verbatim — the same
 *  strings the backend injects, so the promise on the profile and the
 *  instruction the model receives can never drift apart. Each one gets its
 *  own tile: a shield, the rule, and a quiet "Locked" note. */
export function ExpertProtectedRules({ rules }: Props) {
  if (rules.length === 0) return null;

  return (
    <ExpertSection
      title="Rules this expert cannot break"
      description="Part of every expert's soul, and not editable by anyone."
    >
      <ul className="grid gap-3 sm:grid-cols-2">
        {rules.map((rule) => (
          <li
            key={rule}
            className="flex gap-3.5 rounded-2xl bg-white p-4 smooth-shadow-ring-sm"
          >
            <span
              aria-hidden="true"
              className="flex size-9 shrink-0 items-center justify-center rounded-xl bg-gradient-to-b from-emerald-50 to-emerald-100/70 text-emerald-600 ring-1 ring-inset ring-emerald-200/70"
            >
              <Icon icon={SecurityCheckIcon} size={18} />
            </span>
            <div className="min-w-0">
              <p className="text-[15px] leading-6 text-zinc-800">{rule}</p>
              <span className="mt-1 inline-flex items-center gap-1 text-xs text-zinc-500">
                <Icon icon={LockIcon} size={12} aria-hidden="true" />
                Locked
              </span>
            </div>
          </li>
        ))}
      </ul>
    </ExpertSection>
  );
}
