import type { RecipientOption } from "@/app/(platform)/copilot/components/ChatInput/components/RecipientChip";
import type { ExpertIdentity } from "@/app/(platform)/copilot/useExpertMap";

export const OTTO_OPTION: RecipientOption = {
  id: null,
  name: "Otto",
  avatarUrl: null,
};

export function assigneeOptions(
  activeExperts: ExpertIdentity[],
): RecipientOption[] {
  return [
    OTTO_OPTION,
    ...activeExperts.map((expert) => ({
      id: expert.id,
      name: expert.name,
      avatarUrl: expert.avatarUrl,
      color: expert.color,
    })),
  ];
}
