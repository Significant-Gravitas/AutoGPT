import { Text } from "@/components/atoms/Text/Text";
import { InboxIcon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

export function EmptySubmissions() {
  return (
    <div className="flex min-h-[400px] flex-col items-center justify-center rounded-lg border border-border-base bg-background">
      <div className="flex flex-col items-center gap-4 text-center">
        <div className="rounded-full bg-surface-hover p-3">
          <Icon icon={InboxIcon} className="h-8 w-8 text-foreground-tertiary" />
        </div>
        <div className="space-y-2">
          <Text variant="large-medium" className="text-foreground-strong">
            No workflows submitted yet
          </Text>
          <Text variant="body" className="text-foreground-secondary">
            You haven&apos;t submitted any workflows to the store yet.
            <br />
            Click &ldquo;Submit workflow&rdquo; above to get started.
          </Text>
        </div>
      </div>
    </div>
  );
}
