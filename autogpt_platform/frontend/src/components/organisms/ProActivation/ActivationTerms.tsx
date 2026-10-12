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
    <div className="space-y-5">
      <Text variant="body" className="leading-relaxed text-zinc-600">
        {trialEndNotice(terms)}
      </Text>
      <ActivationSummary terms={terms} />
      <ActivationTimeline terms={terms} />
      <RenewalDetails terms={terms} />
    </div>
  );
}
