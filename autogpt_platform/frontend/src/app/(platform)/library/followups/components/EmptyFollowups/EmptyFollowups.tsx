import { Text } from "@/components/atoms/Text/Text";
import { ClockClockwiseIcon } from "@phosphor-icons/react";

export function EmptyFollowups() {
  return (
    <div
      className="flex flex-col items-center justify-center gap-3 rounded-large border border-dashed border-zinc-200 px-6 py-16 text-center"
      data-testid="followups-empty"
    >
      <div className="flex h-12 w-12 items-center justify-center rounded-full bg-yellow-50">
        <ClockClockwiseIcon size={24} className="text-yellow-700" />
      </div>
      <Text variant="h4" className="text-zinc-900">
        Nothing scheduled
      </Text>
      <Text variant="body" className="max-w-md !text-zinc-500">
        Ask your copilot to schedule a follow-up, or add a recurring schedule to
        an agent from the builder — both will show up here so you can review or
        cancel them.
      </Text>
    </div>
  );
}
