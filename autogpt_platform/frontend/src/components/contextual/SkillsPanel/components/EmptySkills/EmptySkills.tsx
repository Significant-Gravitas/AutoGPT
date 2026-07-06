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
        Your AutoPilot distills reusable procedures from past sessions. Hit{" "}
        <strong>New skill</strong> to teach it one in chat, or{" "}
        <strong>Upload skill</strong> to import one you&apos;ve saved — then
        it&apos;ll reach for it automatically.
      </Text>
    </div>
  );
}
