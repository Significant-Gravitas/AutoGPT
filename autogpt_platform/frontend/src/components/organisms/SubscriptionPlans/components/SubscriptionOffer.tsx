import { Tick02Icon } from "@hugeicons/core-free-icons";
import type { TrialOfferResponse } from "@/app/api/__generated__/models/trialOfferResponse";
import { Card } from "@/components/atoms/Card/Card";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import type { PlanDef } from "@/components/molecules/PlanCard/plans";
import { cn } from "@/lib/utils";
import {
  getPlanPresentation,
  isTeamPlan,
  type SubscriptionPlansProps,
} from "../helpers";
import { OfferActions } from "./OfferActions";

interface Props extends SubscriptionPlansProps {
  plan: PlanDef;
  onTrialDetails: () => void;
}

export function SubscriptionOffer({ plan, onTrialDetails, ...props }: Props) {
  const presentation = getPlanPresentation(plan, props);
  return (
    <section aria-label={`${plan.name} plan`} className="h-full">
      <Card
        className={cn(
          "flex h-full flex-col rounded-2xl border border-border bg-card p-5 shadow-none",
          (presentation.trial || plan.highlighted) && "border-accent/40",
        )}
      >
        <OfferHeading plan={plan} trial={presentation.trial} />
        <OfferPrice presentation={presentation} />
        <Text variant="body" tone="primary" className="min-h-11">
          {plan.description}
        </Text>
        <OfferFeatures features={plan.features} isTeam={isTeamPlan(plan.key)} />
        <OfferActions {...props} plan={plan} onTrialDetails={onTrialDetails} />
      </Card>
    </section>
  );
}

function OfferHeading({
  plan,
  trial,
}: {
  plan: PlanDef;
  trial: TrialOfferResponse | null;
}) {
  return (
    <div className="mb-3 flex flex-wrap items-center gap-2">
      <Text variant="h5" as="h2" className="font-semibold text-foreground">
        {plan.name}
      </Text>
      {plan.usage && (
        <span className="rounded-full bg-accent/15 px-2 py-0.5 text-xs text-accent">
          {plan.usage} usage
        </span>
      )}
      {plan.badge && (
        <span
          className={cn(
            "rounded-full px-2 py-0.5 text-xs",
            plan.highlighted
              ? "bg-accent/10 text-accent"
              : "bg-muted text-muted-foreground",
          )}
        >
          {plan.badge}
        </span>
      )}
      {trial && (
        <span className="ml-auto rounded-full bg-accent/10 px-2 py-0.5 text-xs font-medium text-accent">
          {trial.duration_days} days free
        </span>
      )}
    </div>
  );
}

function OfferPrice({
  presentation,
}: {
  presentation: ReturnType<typeof getPlanPresentation>;
}) {
  return (
    <>
      <div className="mb-1 flex flex-wrap items-baseline gap-x-1.5">
        <Text
          variant="h3"
          as="span"
          aria-label={presentation.price}
          unmask={false}
          className="leading-9 text-foreground"
        >
          {presentation.price}
        </Text>
        {presentation.unit && (
          <Text variant="small" as="span" tone="muted">
            {presentation.unit}
          </Text>
        )}
      </div>
      <Text
        variant="small"
        tone={presentation.trial ? undefined : "muted"}
        className={cn("mb-2 min-h-4", presentation.trial && "text-accent")}
      >
        {presentation.caption || "\u00a0"}
      </Text>
    </>
  );
}

function OfferFeatures({
  features,
  isTeam,
}: {
  features: string[];
  isTeam: boolean;
}) {
  return (
    <ul className="mt-3 mb-4 flex-1 space-y-1.5 border-t border-border pt-3">
      {features.map((feature) => (
        <li key={feature} className="flex items-start gap-2">
          <span
            className={cn(
              "mt-0.5 flex h-4 w-4 shrink-0 items-center justify-center rounded-full",
              isTeam ? "bg-muted" : "bg-accent/10",
            )}
          >
            <Icon
              icon={Tick02Icon}
              size={10}
              aria-hidden
              className={isTeam ? "text-muted-foreground" : "text-accent"}
            />
          </span>
          <Text
            variant="small"
            as="span"
            tone="primary"
            className="text-[13px] leading-[18px]"
          >
            {feature}
          </Text>
        </li>
      ))}
    </ul>
  );
}
