import { Text } from "@/components/atoms/Text/Text";
import { BookOpen01Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

export function EmptySkills() {
  return (
    <div
      className="flex flex-col items-center justify-center gap-3 rounded-xl border border-dashed border-zinc-200 px-6 py-16 text-center"
      data-testid="skills-empty"
    >
      <div className="flex h-12 w-12 items-center justify-center rounded-full bg-purple-50">
        <Icon icon={BookOpen01Icon} size={24} className="text-purple-700" />
      </div>
      <Text variant="h4" tone="primary">
        No skills yet
      </Text>
      <Text variant="body" tone="muted" className="max-w-md">
        Give your experts a repeatable process to follow. Select{" "}
        <strong>New skill</strong> to create one in chat, or{" "}
        <strong>Upload skill</strong> to import one you&apos;ve saved.
      </Text>
    </div>
  );
}
