import { Button } from "@/components/atoms/Button/Button";
import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { Text } from "@/components/atoms/Text/Text";
import { PageCard } from "./PageCard";
import { StillClosedHelp } from "./StillClosedHelp";

interface Props {
  isReady: boolean;
  isLeaving: boolean;
  onContinue: () => void;
}

export function OpenAgain({ isReady, isLeaving, onContinue }: Props) {
  return (
    <PageCard title="Your account is open again">
      <Text variant="body">
        Your plan, connected apps, API keys, schedules, share links, and
        Marketplace listings and reviews stay off: subscribe again and set up
        what you need. Notification emails stay off until you switch them back
        on in Settings.
      </Text>
      <StillClosedHelp />
      <div className="flex items-center justify-end gap-3">
        {isReady ? (
          <Button type="button" onClick={onContinue} loading={isLeaving}>
            Continue to AutoGPT
          </Button>
        ) : (
          <>
            <Text variant="small" tone="muted" role="status">
              Opening your account everywhere…
            </Text>
            <LoadingSpinner size="small" />
          </>
        )}
      </div>
    </PageCard>
  );
}
