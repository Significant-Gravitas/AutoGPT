import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { ArrowUpRight01Icon, Message01Icon } from "@hugeicons/core-free-icons";
import NextLink from "next/link";
import { chatHref } from "../../../../helpers";

interface Props {
  sessionIDs: string[];
}

/** The chats the task was worked on in, each opening in the chat view. */
export function TaskChats({ sessionIDs }: Props) {
  if (sessionIDs.length === 0) {
    return (
      <Text variant="inbox" tone="faint">
        No chats yet
      </Text>
    );
  }

  return (
    <ul className="flex flex-col gap-0.5">
      {sessionIDs.map((id, index) => (
        <li key={id}>
          <NextLink
            href={chatHref(id)}
            className="-mx-1.5 flex h-8 items-center gap-2 rounded-control px-1.5 transition-colors duration-150 ease-out-expo hover:bg-surface-wash focus-visible:ring-2 focus-visible:ring-focus-inbox focus-visible:outline-none"
          >
            <Icon icon={Message01Icon} size={14} tone="quiet" />
            <Text as="span" variant="inbox" tone="ink" className="flex-1">
              {sessionIDs.length === 1 ? "Open chat" : `Chat ${index + 1}`}
            </Text>
            <Icon icon={ArrowUpRight01Icon} size={14} tone="faint" />
          </NextLink>
        </li>
      ))}
    </ul>
  );
}
