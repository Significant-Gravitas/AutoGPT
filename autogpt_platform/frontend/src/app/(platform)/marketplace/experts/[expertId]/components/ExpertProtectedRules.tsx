import { Icon } from "@/components/atoms/Icon/Icon";
import { LockIcon } from "@hugeicons/core-free-icons";
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
      <ul className="flex flex-col gap-2">
        {rules.map((rule) => (
          <li
            key={rule}
            className="flex gap-2.5 text-[15px] leading-6 text-zinc-600"
          >
            <Icon
              icon={LockIcon}
              size={16}
              aria-hidden="true"
              className="mt-1 shrink-0 text-zinc-400"
            />
            <span>{rule}</span>
          </li>
        ))}
      </ul>
    </ExpertSection>
  );
}
