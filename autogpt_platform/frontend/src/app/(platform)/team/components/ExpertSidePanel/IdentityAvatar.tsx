import { AutopilotAvatar } from "@/components/molecules/AutopilotAvatar/AutopilotAvatar";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";

export interface PanelIdentity {
  name: string;
  avatarUrl: string | null;
  color?: string | null;
  isAutopilot?: boolean;
}

interface Props {
  identity: PanelIdentity;
  className?: string;
  imageSize: number;
}

export function IdentityAvatar({ identity, className, imageSize }: Props) {
  if (identity.isAutopilot)
    return <AutopilotAvatar size={imageSize} className={className} />;
  return (
    <ExpertAvatar
      name={identity.name}
      avatarUrl={identity.avatarUrl}
      size={imageSize}
      className={className}
    />
  );
}
