import { Icon } from "@/components/atoms/Icon/Icon";
import { LockIcon } from "@hugeicons/core-free-icons";

interface Props {
  rules: string[];
}

/** The rules baked into every expert's soul, quoted verbatim — the same
 *  strings the backend injects, so the promise on the profile and the
 *  instruction the model receives can never drift apart. */
export function ExpertProtectedRules({ rules }: Props) {
  if (rules.length === 0) return null;

  return (
    <section className="rounded-xl bg-zinc-50 px-4 py-3.5">
      <h2 className="flex items-center gap-2 text-sm font-medium text-zinc-900">
        <Icon icon={LockIcon} size={16} className="text-zinc-500" />
        Rules this expert cannot break
      </h2>
      <ul className="mt-2 space-y-1.5">
        {rules.map((rule) => (
          <li
            key={rule}
            className="flex gap-2 text-[13px] leading-5 text-zinc-600"
          >
            <Icon
              icon={LockIcon}
              size={14}
              className="mt-0.5 shrink-0 text-zinc-400"
            />
            <span>{rule}</span>
          </li>
        ))}
      </ul>
    </section>
  );
}
