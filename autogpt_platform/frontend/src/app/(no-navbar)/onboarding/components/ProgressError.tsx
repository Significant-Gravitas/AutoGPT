import { Button } from "@/components/atoms/Button/Button";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";

interface Props {
  message: string;
  conflict: boolean;
  retry: () => void;
}

export function ProgressError({ message, conflict, retry }: Props) {
  return (
    <div className="flex flex-col items-center gap-3">
      <ErrorCard
        context="your onboarding progress"
        responseError={{ message }}
        isOurProblem={!conflict}
        onRetry={retry}
      />
      {conflict && <Button onClick={retry}>Reload latest progress</Button>}
    </div>
  );
}
