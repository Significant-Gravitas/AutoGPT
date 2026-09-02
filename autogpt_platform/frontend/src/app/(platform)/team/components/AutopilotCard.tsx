import { AutoGPTLogo } from "@/components/atoms/AutoGPTLogo/AutoGPTLogo";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import {
  Alert02Icon,
  Clock01Icon,
  Message01Icon,
} from "@hugeicons/core-free-icons";
import { ACTION_BUTTON_CLASS } from "../helpers";

interface Props {
  nextRun: { name: string; when: string } | null;
  attentionCount: number;
}

// Autopilot is not a hired expert, so it deliberately breaks the expert-card
// mould: a tinted panel, an icon instead of a generated marble, and team-wide
// totals rather than one expert's numbers.
export function AutopilotCard({ nextRun, attentionCount }: Props) {
  return (
    <section
      aria-label="Autopilot"
      className="flex flex-col gap-4 rounded-[1.75rem] bg-gradient-to-br from-zinc-100 to-zinc-50 p-5 shadow-zinc-950 smooth-shadow-ring-sm"
    >
      <div className="flex items-start gap-3">
        <span className="flex size-12 shrink-0 items-center justify-center rounded-2xl bg-gradient-to-b from-white to-zinc-100 shadow-[inset_0_1px_1px_rgba(255,255,255,0.9),inset_0_-2px_4px_rgba(0,0,0,0.08),0_1px_2px_rgba(0,0,0,0.06)] ring-1 ring-inset ring-zinc-200/70">
          <AutoGPTLogo hideText viewBox="47 -1 42 42" className="size-6" />
        </span>
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <Text variant="large-medium">Autopilot</Text>
            <span className="rounded-full bg-white px-2 py-0.5 text-xs text-zinc-600 ring-1 ring-inset ring-zinc-200">
              Always on
            </span>
          </div>
          <Text variant="small" className="text-zinc-500">
            Head of AI
          </Text>
        </div>
      </div>

      <Text variant="body" className="text-zinc-600">
        Your built-in generalist. It runs your workflows, answers questions, and
        hands work to your hired experts.
      </Text>

      <dl className="flex flex-col divide-y divide-zinc-200/70 rounded-2xl bg-white/70 px-3">
        <Row label="Next up" icon={Clock01Icon}>
          {nextRun ? `${nextRun.name} · ${nextRun.when}` : "Nothing scheduled"}
        </Row>
        <Row label="Needs you" icon={Alert02Icon}>
          {attentionCount > 0
            ? `${attentionCount} ${attentionCount === 1 ? "item" : "items"}`
            : "All clear"}
        </Row>
      </dl>

      <div className="mt-auto flex gap-2">
        <Button
          as="NextLink"
          href="/copilot"
          variant="primary"
          size="small"
          className={ACTION_BUTTON_CLASS}
          leftIcon={<Icon icon={Message01Icon} size={16} />}
        >
          Chat
        </Button>
      </div>
    </section>
  );
}

function Row({
  label,
  icon,
  children,
}: {
  label: string;
  icon: React.ComponentProps<typeof Icon>["icon"];
  children: React.ReactNode;
}) {
  return (
    <div className="flex items-center justify-between gap-3 py-2.5">
      <dt className="flex shrink-0 items-center gap-1.5">
        <Icon icon={icon} size={14} className="text-zinc-400" />
        <Text variant="small" className="text-zinc-500">
          {label}
        </Text>
      </dt>
      <dd className="min-w-0">
        <Text
          variant="small-medium"
          unmask={false}
          className="truncate text-right text-zinc-900"
        >
          {children}
        </Text>
      </dd>
    </div>
  );
}
