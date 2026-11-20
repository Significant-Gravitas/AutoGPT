import type { ActivationTerms as Terms } from "@/app/api/__generated__/models/activationTerms";
import { Text } from "@/components/atoms/Text/Text";
import { trialEndNotice } from "@/services/pro-activation/reviewDetails";
import { ActivationSummary } from "./ActivationSummary";
import { ActivationTimeline } from "./ActivationTimeline";
import { RenewalDetails } from "./RenewalDetails";

interface Props {
  terms: Terms;
}

export function ActivationTerms({ terms }: Props) {
  return (
    <div data-slot="activation-terms" className="space-y-5">
      <Text variant="body" tone="secondary" className="leading-relaxed">
        {trialEndNotice(terms)}
      </Text>
      <ActivationSummary terms={terms} />
      <ActivationTimeline terms={terms} />
      <RenewalDetails terms={terms} />
    </div>
  );
}
