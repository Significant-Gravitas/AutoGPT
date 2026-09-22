import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { AuthCard } from "@/components/auth/AuthCard";
import { LinkType } from "@/app/api/__generated__/models/linkType";
import { isUserLink } from "../helpers";

interface Props {
  linkType: LinkType;
  platform: string;
  serverName: string | null;
  serverNoun: string;
  userEmail: string | null;
  isLinking: boolean;
  onLink: () => void;
  onSwitchAccount: () => void;
}

export function ReadyView({
  linkType,
  platform,
  serverName,
  serverNoun,
  userEmail,
  isLinking,
  onLink,
  onSwitchAccount,
}: Props) {
  const forUser = isUserLink(linkType);
  const title = buildTitle({ forUser, platform, serverName, serverNoun });
  const contextLabel = forUser
    ? `your ${platform} DMs`
    : (serverName ?? `this ${platform} ${serverNoun}`);

  return (
    <AuthCard title={title} className="max-w-[38rem] gap-6 p-8">
      <div className="flex w-full flex-col items-center gap-6">
        <div className="w-full rounded-xl bg-muted px-6 py-5 text-left">
          <Text variant="body-medium" className="font-medium">
            What happens when you confirm:
          </Text>
          {forUser ? (
            <ul className="mt-4 space-y-2.5 text-sm leading-relaxed text-muted-foreground">
              <li>{contextLabel} will be linked to your AutoGPT account</li>
              <li>DMs with the bot run as your own private AutoGPT chat</li>
              <li>All usage from those DMs is billed to your account</li>
            </ul>
          ) : (
            <ul className="mt-4 space-y-2.5 text-sm leading-relaxed text-muted-foreground">
              <li>{contextLabel} will be connected to your AutoGPT account</li>
              <li>Anyone in the {serverNoun} can give AutoGPT tasks</li>
              <li>
                Conversations in a shared {serverNoun} are visible to everyone
                there — for private 1:1 work, DM the bot and link your own
                AutoGPT account
              </li>
              <li>
                Uses the tools and integrations connected to your AutoGPT
                account
              </li>
            </ul>
          )}
        </div>

        <div className="w-full rounded-xl border border-border bg-muted px-5 py-4">
          <Text
            variant="small"
            className="leading-relaxed text-muted-foreground"
          >
            Usage from {contextLabel} is billed to your AutoGPT account. You can
            unlink at any time in Settings → Bots.
          </Text>
        </div>

        <Button
          onClick={onLink}
          loading={isLinking}
          disabled={isLinking}
          className="w-full sm:w-auto sm:min-w-[16rem]"
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
  serverNoun: string;
}): string {
  if (args.forUser) return `Link your ${args.platform} DMs`;
  if (args.serverName) return `Set up AutoGPT for ${args.serverName}`;
  return `Set up AutoGPT for this ${args.platform} ${args.serverNoun}`;
}
