import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { WizardPlaceholder } from "../WizardPlaceholder";
import { useLoadingSetup } from "./useLoadingSetup";

interface Props {
  totalSteps: number;
}

export function LoadingSetup({ totalSteps }: Props) {
  const { isStalled, reload } = useLoadingSetup();

  if (isStalled)
    return (
      <ErrorCard
        context="your setup"
        responseError={{
          message:
            "Your setup is taking too long to load. Check your connection, then try again.",
        }}
        onRetry={reload}
      />
    );

  return (
    <WizardPlaceholder message="Loading your setup…" totalSteps={totalSteps} />
  );
}
