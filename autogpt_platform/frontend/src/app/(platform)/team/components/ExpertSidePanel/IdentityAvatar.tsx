import { AutopilotAvatar } from "@/components/molecules/AutopilotAvatar/AutopilotAvatar";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import { cn } from "@/lib/utils";

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
  // TODO(tokens): no exact role: border-[#e3e3e3] #E3E3E3
  const avatarClassName = cn("rounded-full border border-[#e3e3e3]", className);
  if (identity.isAutopilot)
    return <AutopilotAvatar size={imageSize} className={avatarClassName} />;
  return (
    <ExpertAvatar
      name={identity.name}
      avatarUrl={identity.avatarUrl}
      size={imageSize}
      className={avatarClassName}
    />
  );
}
