import { Text } from "@/components/atoms/Text/Text";
import { Button } from "@/components/atoms/Button/Button";
import { InboxIcon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

export function SubmissionLoadError() {
  return (
    <div className="flex min-h-[400px] flex-col items-center justify-center rounded-lg border border-neutral-200 bg-neutral-50">
      <div className="flex flex-col items-center gap-4 text-center">
        <div className="rounded-full bg-red-100 p-3">
          <Icon icon={InboxIcon} className="h-8 w-8 text-red-600" />
        </div>
        <div className="space-y-2">
          <Text variant="large-medium" className="text-neutral-900">
            Failed to load workflows
          </Text>
          <Text variant="body" className="text-neutral-600">
            Something went wrong while loading your submitted workflows.
          </Text>
        </div>
        <Button
          variant="secondary"
          size="small"
          onClick={() => window.location.reload()}
        >
          Try again
        </Button>
      </div>
    </div>
  );
}
