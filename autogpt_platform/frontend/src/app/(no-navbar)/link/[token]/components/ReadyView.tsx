import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { AuthCard } from "@/components/auth/AuthCard";
import { LinkType } from "@/app/api/__generated__/models/linkType";
import { isUserLink } from "../helpers";

interface Props {
  linkType: LinkType;
  platform: string;
  serverName: string | null;
  userEmail: string | null;
  isLinking: boolean;
  onLink: () => void;
  onSwitchAccount: () => void;
}

export function ReadyView({
  linkType,
  platform,
  serverName,
  userEmail,
  isLinking,
  onLink,
  onSwitchAccount,
}: Props) {
  const forUser = isUserLink(linkType);
  const title = buildTitle({ forUser, platform, serverName });
  const contextLabel = forUser
    ? `your ${platform} DMs`
    : (serverName ?? `this ${platform} server`);

  return (
    <AuthCard title={title}>
      <div className="flex w-full flex-col items-center gap-6">
        <div className="w-full rounded-xl bg-muted p-5 text-left">
          <Text variant="body-medium" className="font-medium">
            What happens when you confirm:
          </Text>
          {forUser ? (
            <ul className="mt-3 space-y-2 text-sm text-muted-foreground">
              <li>{contextLabel} will be linked to your AutoGPT account</li>
              <li>DMs with the bot run as your personal AutoPilot</li>
              <li>All usage from those DMs is billed to your account</li>
            </ul>
          ) : (
            <ul className="mt-3 space-y-2 text-sm text-muted-foreground">
              <li>{contextLabel} will be connected to your AutoGPT account</li>
              <li>Everyone in the server can chat with AutoPilot</li>
              <li>Each person gets their own private conversation</li>
              <li>
                All usage from the server is billed to your AutoGPT account
              </li>
            </ul>
          )}
        </div>

        <div className="w-full rounded-xl border border-border bg-muted p-4">
          <Text variant="small" className="text-muted-foreground">
            Usage from {contextLabel} will be billed to your AutoGPT account.
            You can unlink at any time from your account settings.
          </Text>
        </div>

        <Button
          onClick={onLink}
          loading={isLinking}
          disabled={isLinking}
          className="w-full"
        >
          {forUser
            ? `Connect my ${platform} DMs`
            : `Connect ${platform} to AutoGPT`}
        </Button>

        {userEmail && (
          <div className="flex w-full items-center justify-between">
            <Text variant="small" className="text-muted-foreground">
              Signed in as {userEmail}
            </Text>
            <Button
              variant="ghost"
              size="small"
              onClick={onSwitchAccount}
              className="text-xs text-muted-foreground underline underline-offset-2"
            >
              Not you? Sign out
            </Button>
          </div>
        )}
      </div>
    </AuthCard>
  );
}

function buildTitle(args: {
  forUser: boolean;
  platform: string;
  serverName: string | null;
}): string {
  if (args.forUser) return `Link your ${args.platform} DMs`;
  if (args.serverName) return `Set up AutoPilot for ${args.serverName}`;
  return `Set up AutoPilot for this ${args.platform} server`;
}
