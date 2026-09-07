import { Text } from "@/components/atoms/Text/Text";

export function PersonalExpertWorkspaceNotice() {
  return (
    <div
      role="status"
      className="space-y-2 rounded-xl border border-zinc-200 bg-white p-5"
    >
      <Text variant="body-medium">
        Experts currently belong to your personal workspace
      </Text>
      <Text variant="body" tone="muted">
        Switch to your own personal workspace to hire or create an expert.
        Shared workspace experts are not available yet.
      </Text>
    </div>
  );
}
