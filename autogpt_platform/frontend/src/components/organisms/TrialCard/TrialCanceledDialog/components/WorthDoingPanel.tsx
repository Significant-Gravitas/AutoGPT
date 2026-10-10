import {
  Clock01Icon,
  DollarSignIcon,
  Note01Icon,
} from "@hugeicons/core-free-icons";
import type { TrialOfferResponse } from "@/app/api/__generated__/models/trialOfferResponse";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import {
  describeTrialTimeLeft,
  formatPlanPrice,
  formatTrialEndDate,
  trialPlanLabels,
} from "../../helpers";

interface Props {
  offer: TrialOfferResponse;
  endsAt: Date | string | null | undefined;
}

export function WorthDoingPanel({ offer, endsAt }: Props) {
  const planStart = describePlanStart(endsAt);
  const resumeNote = planStart
    ? ` Resume your trial instead and your plan starts ${planStart}.`
    : "";
  const rows = [
    {
      icon: Clock01Icon,
      title: "Put an Expert on a schedule",
      body: 'Say "do this every Monday at 8am" to any Expert. It runs while you\'re away and the result is waiting on Home.',
    },
    {
      icon: Note01Icon,
      title: "Your work stays",
      body: "Chats, Experts and schedules stay on your account. Subscribe later and you pick up where you left off.",
    },
    {
      icon: DollarSignIcon,
      title: "Subscribe any time",
      body: `${trialPlanLabels[offer.tier]} is ${formatPlanPrice(offer)}, cancel anytime.${resumeNote}`,
    },
  ];

  return (
    <div className="rounded-xl border border-zinc-200 bg-zinc-50 px-4 pt-3">
      <Text variant="eyebrow" as="h3">
        Worth doing before then
      </Text>
      <ul className="divide-y divide-zinc-200">
        {rows.map((row) => (
          <li key={row.title} className="flex gap-3 py-3">
            <span className="flex size-7 shrink-0 items-center justify-center rounded-full bg-violet-100">
              <Icon icon={row.icon} size={14} className="text-violet-700" />
            </span>
            <div className="flex min-w-0 flex-col gap-0.5">
              <Text variant="body-medium" className="!text-zinc-900">
                {row.title}
              </Text>
              <Text variant="small" unmask={false} className="!text-zinc-600">
                {row.body}
              </Text>
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}

function describePlanStart(endsAt: Props["endsAt"]) {
  if (!endsAt) return null;
  const left = describeTrialTimeLeft(endsAt);
  if (left.kind === "ended") return null;
  return left.kind === "days" ? formatTrialEndDate(endsAt) : left.kind;
}
