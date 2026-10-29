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
  const avatarClassName = cn(
    // oxlint-disable-next-line design-system/no-restyle -- ExpertAvatar and AutopilotAvatar have no bordered variant (src/components is P9's)
    "rounded-full border border-border-base",
    className,
  );
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
