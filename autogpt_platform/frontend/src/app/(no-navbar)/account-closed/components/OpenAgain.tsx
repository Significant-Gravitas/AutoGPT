import { Button } from "@/components/atoms/Button/Button";
import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { Text } from "@/components/atoms/Text/Text";
import { PageCard } from "./PageCard";
import { StillClosedHelp } from "./StillClosedHelp";
import { useOpenAgain } from "./useOpenAgain";

interface Props {
  isReady: boolean;
  isLeaving: boolean;
  onContinue: () => void;
}

export function OpenAgain({ isReady, isLeaving, onContinue }: Props) {
  const { headingID, continueID, statusText } = useOpenAgain(isReady);

  return (
    <PageCard title="Your account is open again" headingID={headingID}>
      <Text variant="body">
        Your plan, connected apps, API keys, schedules and share links stay off,
        and your Marketplace listings and reviews are removed: subscribe again
        and set up what you need. Notification emails stay off until you switch
        them back on in Settings.
      </Text>
      <StillClosedHelp />
      <div className="flex items-center justify-end gap-3">
        {/* One live region for the whole wait, so its change is announced. */}
        <Text variant="small" tone="muted" role="status" aria-live="polite">
          {statusText}
        </Text>
        {isReady ? (
          <Button
            id={continueID}
            type="button"
            onClick={onContinue}
            loading={isLeaving}
          >
            Continue to AutoGPT
          </Button>
        ) : (
          <LoadingSpinner size="small" aria-hidden />
        )}
      </div>
    </PageCard>
  );
}
