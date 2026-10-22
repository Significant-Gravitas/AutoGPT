import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import type { PlanDef } from "@/components/molecules/PlanCard/plans";
import { formatTrialPrice } from "@/components/organisms/TrialCard/helpers";
import {
  getPlanPresentation,
  isTeamPlan,
  type SubscriptionPlansProps,
} from "../helpers";

interface Props extends SubscriptionPlansProps {
  plan: PlanDef;
  onTrialDetails: () => void;
}

export function OfferActions(props: Props) {
  const { trial, offer, action } = getPlanPresentation(props.plan, props);
  return (
    <div className="mt-auto space-y-2">
      {trial ? (
        <Button
          type="button"
          variant="primary"
          size="small"
          className="w-full"
          loading={props.isStartingTrial}
          disabled={props.isStartingTrial || props.isUpdatingTier}
          onClick={props.onStartTrial}
        >
          {action}
        </Button>
      ) : (
        <PaidPlanAction {...props} />
      )}
      <div className="min-h-10 text-center">
        {trial ? (
          <TrialActionDetails {...props} />
        ) : offer ? (
          <AlternativeTrial {...props} offer={offer} />
        ) : (
          <Text variant="small" tone="muted">
            {isTeamPlan(props.plan.key)
              ? "Find the right fit for your team."
              : "Manage your plan and billing anytime."}
          </Text>
        )}
      </div>
      {offer && props.trialError && (
        <p role="alert" className="text-sm text-destructive">
          {props.trialError}
        </p>
      )}
    </div>
  );
}

function PaidPlanAction({
  plan,
  isUpdatingTier,
  isStartingTrial,
  selectedPlan,
  onSelectPlan,
  inline = false,
  price,
}: Props & { inline?: boolean; price?: string | null }) {
  function handleSelect() {
    onSelectPlan(plan.key);
  }

  return (
    <Button
      type="button"
      variant={inline ? "link" : plan.highlighted ? "primary" : "secondary"}
      size="small"
      className={inline ? "h-auto min-w-0 p-0 text-xs text-zinc-500" : "w-full"}
      loading={isUpdatingTier && selectedPlan === plan.key}
      disabled={isUpdatingTier || isStartingTrial}
      onClick={handleSelect}
    >
      {plan.cta}
      {price && ` · ${price}`}
    </Button>
  );
}

function TrialActionDetails(props: Props) {
  const { paidActionPrice } = getPlanPresentation(props.plan, props);
  return (
    <Text variant="small" tone="muted">
      Card required. No subscription charge today.{" "}
      <Button
        type="button"
        variant="link"
        className="h-auto min-w-0 p-0 text-xs text-zinc-500"
        onClick={props.onTrialDetails}
      >
        Trial details
      </Button>
      {" · "}
      <PaidPlanAction {...props} inline price={paidActionPrice} />
    </Text>
  );
}

function AlternativeTrial({
  plan,
  offer,
  onBillingChange,
  isStartingTrial,
}: Props & {
  offer: NonNullable<SubscriptionPlansProps["trialOffer"]>;
}) {
  function handleSwitchCycle() {
    onBillingChange(offer.billing_cycle);
  }

  return (
    <div>
      <Button
        type="button"
        variant="link"
        className="h-auto min-w-0 p-0 text-xs text-purple-500"
        disabled={isStartingTrial}
        onClick={handleSwitchCycle}
      >
        Try {plan.name} for {offer.duration_days} days instead
      </Button>
      <Text variant="small" tone="muted">
        Renews at {formatTrialPrice(offer)} + applicable tax.
      </Text>
    </div>
  );
}
