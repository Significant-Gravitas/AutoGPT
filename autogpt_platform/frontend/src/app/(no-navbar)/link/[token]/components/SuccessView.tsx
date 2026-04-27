import { Text } from "@/components/atoms/Text/Text";
import { AuthCard } from "@/components/auth/AuthCard";
import { LinkType } from "@/app/api/__generated__/models/linkType";
import { CheckCircle } from "@phosphor-icons/react";
import { isUserLink } from "../helpers";

interface Props {
  linkType: LinkType;
  platform: string;
  serverName: string | null;
}

export function SuccessView({ linkType, platform, serverName }: Props) {
  const forUser = isUserLink(linkType);
  const label =
    forUser || !serverName ? `your ${platform} account` : serverName;
  const detail = forUser
    ? `You can now chat with AutoPilot in your ${platform} DMs.`
    : `Everyone in the server can start using AutoPilot right away.`;

  return (
    <AuthCard title="AutoPilot is ready!">
      <div className="flex w-full flex-col items-center gap-6">
        <div className="flex h-16 w-16 items-center justify-center rounded-full bg-green-100">
          <CheckCircle size={40} weight="fill" className="text-green-600" />
        </div>
        <Text
          variant="body-medium"
          className="text-center text-muted-foreground"
        >
          <strong>{label}</strong> is now connected to your AutoGPT account.
          <br />
          {detail}
        </Text>
        <Text variant="small" className="text-center text-muted-foreground">
          You can close this page and go back to your chat.
        </Text>
      </div>
    </AuthCard>
  );
}
