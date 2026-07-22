import { Text } from "@/components/atoms/Text/Text";
import { CalendarDotsIcon } from "@phosphor-icons/react";

export function EmptyFollowups() {
  return (
    <div
      className="flex flex-col items-center justify-center gap-3 rounded-large border border-dashed border-zinc-200 px-6 py-16 text-center"
      data-testid="followups-empty"
    >
      <div className="flex h-12 w-12 items-center justify-center rounded-full bg-yellow-50">
        <CalendarDotsIcon size={24} className="text-yellow-700" />
      </div>
      <Text variant="h4" className="text-zinc-900">
        Nothing scheduled yet
      </Text>
      <Text variant="body" className="max-w-md !text-zinc-500">
        Recurring agent runs and your AutoPilot&apos;s follow-up messages show
        up here. Hit <strong>New scheduled task</strong> to set one up, or
        schedule an agent from the builder.
      </Text>
    </div>
  );
}
