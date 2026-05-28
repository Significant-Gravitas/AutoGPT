import { Text } from "@/components/atoms/Text/Text";
import { BookOpenIcon } from "@phosphor-icons/react";

export function EmptySkills() {
  return (
    <div
      className="flex flex-col items-center justify-center gap-3 rounded-large border border-dashed border-zinc-200 px-6 py-16 text-center"
      data-testid="skills-empty"
    >
      <div className="flex h-12 w-12 items-center justify-center rounded-full bg-violet-50">
        <BookOpenIcon size={24} className="text-violet-700" />
      </div>
      <Text variant="h4" className="text-zinc-900">
        No skills yet
      </Text>
      <Text variant="body" className="max-w-md !text-zinc-500">
        Your copilot will distill skills here as it works through tasks worth
        remembering. They appear automatically — no setup needed.
      </Text>
    </div>
  );
}
