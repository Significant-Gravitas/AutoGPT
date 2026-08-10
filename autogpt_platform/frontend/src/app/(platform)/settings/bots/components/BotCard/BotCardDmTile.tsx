import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import type { PlatformUserLinkInfo } from "@/app/api/__generated__/models/platformUserLinkInfo";
import {
  Chatting01Icon,
  Delete02Icon,
  Link01Icon,
} from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

type Props = {
  platformName: string;
  dmLink: PlatformUserLinkInfo | null;
  isPending: (linkId: string) => boolean;
  onUnlink: (linkId: string) => void;
};

export function BotCardDmTile({
  platformName,
  dmLink,
  isPending,
  onUnlink,
}: Props) {
  const title = dmLink
    ? (dmLink.platform_username ?? `User ${dmLink.platform_user_id}`)
    : `DM the bot on ${platformName} to link`;
  const subtitle = dmLink
    ? "Your DM channel is linked to this account."
    : "Send the bot a direct message to start the link flow.";

  return (
    <div className="flex items-center justify-between gap-3 rounded-large border border-zinc-200 px-4 py-3">
      <div className="flex min-w-0 items-center gap-3">
        <Icon
          icon={Chatting01Icon}
          size={20}
          className="shrink-0 text-zinc-500"
        />
        <div className="flex min-w-0 flex-col">
          <Text variant="body-medium" as="span" className="text-textBlack">
            {title}
          </Text>
          <Text variant="small" as="span" className="text-zinc-500">
            {subtitle}
          </Text>
        </div>
      </div>
      {dmLink ? (
        <Button
          variant="outline"
          size="small"
          leftIcon={<Icon icon={Delete02Icon} size={16} />}
          loading={isPending(dmLink.id)}
          onClick={() => onUnlink(dmLink.id)}
          aria-label={`Unlink DM on ${platformName}`}
        >
          Unlink
        </Button>
      ) : (
        <span className="inline-flex items-center gap-1 text-xs text-zinc-500">
          <Icon icon={Link01Icon} size={14} /> Not linked
        </span>
      )}
    </div>
  );
}
