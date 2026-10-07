import { Text } from "@/components/atoms/Text/Text";
import { InboxIcon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

export function EmptySubmissions() {
  return (
    <div className="flex min-h-[400px] flex-col items-center justify-center rounded-lg border border-zinc-200 bg-zinc-50">
      <div className="flex flex-col items-center gap-4 text-center">
        <div className="rounded-full bg-zinc-100 p-3">
          <Icon icon={InboxIcon} size={32} className="text-zinc-500" />
        </div>
        <div className="space-y-2">
          <Text variant="large-medium" tone="primary">
            No agents submitted yet
          </Text>
          <Text variant="body" tone="secondary">
            You haven&apos;t submitted any agents to the store yet.
            <br />
            Click &ldquo;Submit agent&rdquo; above to get started.
          </Text>
        </div>
      </div>
    </div>
  );
}
