import { AUTOPILOT_BLURB } from "../../helpers";

const ENTRIES = [
  {
    label: "Identity",
    value:
      "Autopilot is the generalist at the head of your team. It talks with you directly, answers questions, and runs your workflows on request.",
  },
  {
    label: "Works with",
    value:
      "Every expert you hire. Autopilot knows what each one can do and hands work to the right expert, then reports back to you.",
  },
  {
    label: "Boundaries",
    value:
      "Autopilot asks before taking external actions on your behalf and never edits an expert's Soul for you.",
  },
] as const;

export function AutopilotAboutSection() {
  return (
    <section className="space-y-5">
      <p className="text-base leading-relaxed text-zinc-600">
        {AUTOPILOT_BLURB}
      </p>

      <dl className="space-y-5">
        {ENTRIES.map((entry) => (
          <div key={entry.label}>
            <dt className="text-sm font-medium text-zinc-900">{entry.label}</dt>
            <dd className="mt-1 text-base leading-relaxed text-zinc-600">
              {entry.value}
            </dd>
          </div>
        ))}
      </dl>
    </section>
  );
}
