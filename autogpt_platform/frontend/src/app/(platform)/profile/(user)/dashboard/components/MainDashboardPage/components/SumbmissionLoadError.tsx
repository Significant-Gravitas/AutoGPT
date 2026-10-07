import { Text } from "@/components/atoms/Text/Text";
import { Button } from "@/components/atoms/Button/Button";
import { InboxIcon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

export function SubmissionLoadError() {
  return (
    <div className="flex min-h-[400px] flex-col items-center justify-center rounded-lg border border-zinc-200 bg-zinc-50">
      <div className="flex flex-col items-center gap-4 text-center">
        <div className="rounded-full bg-red-100 p-3">
          <Icon icon={InboxIcon} size={32} className="text-red-600" />
        </div>
        <div className="space-y-2">
          <Text variant="large-medium" tone="primary">
            Failed to load agents
          </Text>
          <Text variant="body" tone="secondary">
            Something went wrong while loading your submitted agents.
          </Text>
        </div>
        <Button
          variant="secondary"
          size="md"
          onClick={() => window.location.reload()}
        >
          Try again
        </Button>
      </div>
    </div>
  );
}
