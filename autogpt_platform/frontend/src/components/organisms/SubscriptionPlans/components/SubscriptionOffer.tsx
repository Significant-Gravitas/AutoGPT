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
    <section
      aria-label={`${plan.name} plan`}
      data-slot="subscription-offer"
      className="h-full"
    >
      <Card
        className={cn(
          "flex h-full flex-col rounded-2xl border border-border-strong bg-surface p-5 shadow-none",
          // TODO(tokens): no role for the purple-300 border.
          (presentation.trial || plan.highlighted) && "border-purple-300",
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
    <div
      data-slot="offer-heading"
      className="mb-3 flex flex-wrap items-center gap-2"
    >
      <Text
        variant="h5"
        as="h2"
        className="font-semibold text-foreground-emphasis"
      >
        {plan.name}
      </Text>
      {plan.usage && (
        <span
          data-slot="offer-heading-usage"
          // TODO(tokens): no role for the purple-100 pill or its purple-700
          // text.
          className="rounded-full bg-purple-100 px-2 py-0.5 text-xs text-purple-700"
        >
          {plan.usage} usage
        </span>
      )}
      {plan.badge && (
        <span
          data-slot="offer-heading-badge"
          className={cn(
            "rounded-full px-2 py-0.5 text-xs",
            plan.highlighted
              ? // TODO(tokens): no role for the purple-50 pill or its purple-700
                // text.
                "bg-purple-50 text-purple-700"
              : "bg-surface-hover text-foreground-tertiary",
          )}
        >
          {plan.badge}
        </span>
      )}
      {trial && (
        <span
          data-slot="offer-heading-trial"
          // TODO(tokens): no role for the purple-50 pill or its purple-700
          // text.
          className="ml-auto rounded-full bg-purple-50 px-2 py-0.5 text-xs font-medium text-purple-700"
        >
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
      <div
        data-slot="offer-price"
        className="mb-1 flex flex-wrap items-baseline gap-x-1.5"
      >
        <Text
          variant="h3"
          as="span"
          aria-label={presentation.price}
          className="leading-9 text-foreground-emphasis"
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
        // TODO(tokens): no text role for purple-700.
        className={cn("mb-2 min-h-4", presentation.trial && "text-purple-700")}
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
    <ul
      data-slot="offer-features"
      className="mb-4 mt-3 flex-1 space-y-1.5 border-t border-separator pt-3"
    >
      {features.map((feature) => (
        <li
          key={feature}
          data-slot="offer-features-item"
          className="flex items-start gap-2"
        >
          <span
            data-slot="offer-features-icon"
            className={cn(
              "mt-0.5 flex h-4 w-4 shrink-0 items-center justify-center rounded-full",
              // TODO(tokens): no role for the purple-50 tick surface.
              isTeam ? "bg-surface-hover" : "bg-purple-50",
            )}
          >
            <Icon
              icon={Tick02Icon}
              size={10}
              aria-hidden
              className={isTeam ? "text-foreground-tertiary" : "text-brand"}
            />
          </span>
          <Text
            variant="small"
            as="span"
            tone="primary"
            className="text-body-sm leading-4.5"
          >
            {feature}
          </Text>
        </li>
      ))}
    </ul>
  );
}
