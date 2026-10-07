import { InformationCircleIcon } from "@hugeicons/core-free-icons";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";

type Props = {
  reason?: string;
  onRetry: () => void;
};

export function SharedChatErrorState({ onRetry }: Props) {
  return (
    <div className="flex h-full w-full flex-1 items-center justify-center">
      <div className="mx-auto w-full max-w-md p-6">
        <div className="space-y-4 rounded-lg border border-dashed border-zinc-300 p-6 text-center">
          <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-full bg-zinc-100">
            <Icon
              icon={InformationCircleIcon}
              size={24}
              className="text-muted-foreground"
            />
          </div>
          <div className="space-y-2">
            <Text variant="large-semibold" as="h3">
              Share link not found
            </Text>
            <Text variant="body" tone="muted">
              This link is invalid or has been disabled by the owner. Ask the
              person who shared it for an updated link.
            </Text>
          </div>
          <Button variant="link" onClick={onRetry}>
            Try again
          </Button>
        </div>
        <Text variant="small" className="mt-8 text-center text-zinc-400">
          Powered by AutoGPT Platform
        </Text>
      </div>
    </div>
  );
}
