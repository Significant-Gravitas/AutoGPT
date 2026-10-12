import type { HomeAttentionItem } from "@/app/api/__generated__/models/homeAttentionItem";
import { AutopilotAvatar } from "@/components/molecules/AutopilotAvatar/AutopilotAvatar";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";

interface Props {
  item: HomeAttentionItem;
  size: number;
}

export function HeldAvatar({ item, size }: Props) {
  return item.expert ? (
    <ExpertAvatar
      name={item.expert.name}
      avatarUrl={item.expert.avatar_url}
      size={size}
    />
  ) : (
    <AutopilotAvatar size={size} />
  );
}
