import { Text } from "@/components/atoms/Text/Text";
import { AUTOPILOT_BLURB } from "../../helpers";

const ENTRIES = [
  { label: "Bio", value: AUTOPILOT_BLURB },
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
      <dl className="space-y-4">
        {ENTRIES.map((entry) => (
          <div key={entry.label}>
            <Text variant="body-medium" as="dt" tone="primary">
              {entry.label}
            </Text>
            <Text variant="body" as="dd" tone="secondary" className="mt-1">
              {entry.value}
            </Text>
          </div>
        ))}
      </dl>
    </section>
  );
}
