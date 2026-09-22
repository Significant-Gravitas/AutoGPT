import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { AuthCard } from "@/components/auth/AuthCard";
import { LinkType } from "@/app/api/__generated__/models/linkType";
import { isUserLink } from "../helpers";
import {
  ArrowTurnBackwardIcon,
  CheckmarkCircle02Icon,
} from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

interface Props {
  linkType: LinkType;
  platform: string;
  serverName: string | null;
  serverNoun: string;
  returnUrl: string | null;
}

export function SuccessView({
  linkType,
  platform,
  serverName,
  serverNoun,
  returnUrl,
}: Props) {
  const forUser = isUserLink(linkType);
  const label =
    forUser || !serverName ? `your ${platform} account` : serverName;
  const detail = forUser
    ? `You can now chat with AutoGPT in your ${platform} DMs.`
    : `Everyone in the ${serverNoun} can start using AutoGPT right away.`;

  return (
    <AuthCard title="AutoGPT is ready!" className="max-w-[38rem] gap-6 p-8">
      <div className="flex w-full flex-col items-center gap-7">
        <div className="flex h-16 w-16 items-center justify-center rounded-full bg-green-100">
          <Icon
            icon={CheckmarkCircle02Icon}
            size={40}
            className="text-green-600"
          />
        </div>

        <div className="w-full rounded-xl bg-muted px-6 py-5 text-center">
          <Text variant="body-medium" className="text-muted-foreground">
            <strong>{label}</strong> is now connected to your AutoGPT account.
            <br />
            {detail}
          </Text>
          {returnUrl && forUser ? (
            <Text variant="small" className="mt-3 block text-muted-foreground">
              Try it now — &ldquo;research a topic&rdquo;, &ldquo;build me an
              agent&rdquo;, or &ldquo;draft a doc&rdquo;.
            </Text>
          ) : null}
        </div>

        <div className="flex w-full flex-col items-stretch gap-3 sm:flex-row sm:justify-center sm:gap-4">
          {returnUrl ? (
            <Button
              as="NextLink"
              href={returnUrl}
              size="small"
              leftIcon={<Icon icon={ArrowTurnBackwardIcon} size={16} />}
            >
              Return to {platform}
            </Button>
          ) : null}
          <Button
            as="NextLink"
            href="/settings/bots"
            variant="outline"
            size="small"
          >
            Manage bots
          </Button>
        </div>

        {returnUrl ? null : (
          <Text variant="small" className="text-center text-muted-foreground">
            You can close this page and go back to your chat.
          </Text>
        )}
      </div>
    </AuthCard>
  );
}
