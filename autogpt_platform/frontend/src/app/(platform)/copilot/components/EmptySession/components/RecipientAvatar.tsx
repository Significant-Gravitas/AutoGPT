import { AutopilotAvatar } from "@/components/molecules/AutopilotAvatar/AutopilotAvatar";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import type { AvatarShape } from "@/components/molecules/ExpertAvatar/helpers";
import type { RecipientOption } from "../../ChatInput/components/RecipientChip";

interface Props {
  recipient: RecipientOption;
  size: number;
  shape: AvatarShape;
}

/** Otto's own mark, or the expert's portrait. */
export function RecipientAvatar({ recipient, size, shape }: Props) {
  if (recipient.id === null && !recipient.avatarUrl) {
    return <AutopilotAvatar size={size} shape={shape} />;
  }
  return (
    <ExpertAvatar
      name={recipient.name}
      avatarUrl={recipient.avatarUrl}
      size={size}
      shape={shape}
    />
  );
}
