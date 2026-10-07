import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { Text } from "@/components/atoms/Text/Text";

export function SharedChatLoadingState() {
  return (
    <div
      data-testid="shared-chat-loading-state"
      className="flex h-full w-full flex-1 items-center justify-center"
    >
      <div className="text-center">
        <LoadingSpinner size="large" className="mx-auto mb-4" />
        <Text variant="body" tone="muted">
          Loading shared chat…
        </Text>
      </div>
    </div>
  );
}
