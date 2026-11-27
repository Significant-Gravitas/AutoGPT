import { InformationCircleIcon } from "@hugeicons/core-free-icons";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";

interface Props {
  isDeadLink: boolean;
  onRetry: () => void;
}

export function SharedChatErrorState({ isDeadLink, onRetry }: Props) {
  return (
    <div className="flex h-full w-full flex-1 items-center justify-center">
      <div className="mx-auto w-full max-w-md p-6">
        <div className="space-y-4 rounded-lg border border-dashed border-border-strong p-6 text-center">
          <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-full bg-surface-hover text-foreground-tertiary">
            <Icon icon={InformationCircleIcon} size={24} aria-hidden />
          </div>
          <div className="space-y-2">
            <Text variant="lead" weight="semibold" as="h3">
              {isDeadLink ? "Share link not found" : "Unable to load"}
            </Text>
            <Text variant="body" tone="muted">
              {isDeadLink
                ? "This link is invalid or has been disabled by the owner. Ask the person who shared it for an updated link."
                : "There was an error loading this shared chat. Please try again."}
            </Text>
          </div>
          {!isDeadLink && (
            <Button variant="link" size="inline" onClick={onRetry}>
              Try again
            </Button>
          )}
        </div>
        <Text variant="small" tone="subtle" align="center" className="mt-8">
          Powered by AutoGPT Platform
        </Text>
      </div>
    </div>
  );
}
