import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { BubbleChatBlockedIcon } from "@hugeicons/core-free-icons";
import { parseAsString, useQueryState } from "nuqs";

export function SessionNotFound() {
  const [, setSessionId] = useQueryState("sessionId", parseAsString);
  const [, setExpertId] = useQueryState("expertId", parseAsString);

  function handleNewChat() {
    setSessionId(null);
    // Otherwise the ?expertId= deep-link adoption drops the user into that
    // expert's latest thread instead of a new chat.
    setExpertId(null);
  }

  return (
    <div className="flex h-full min-h-0 w-full items-center justify-center bg-white px-6">
      <div
        role="status"
        className="flex max-w-md flex-col items-center gap-3 text-center"
      >
        <Icon
          icon={BubbleChatBlockedIcon}
          size={28}
          className="text-muted-foreground"
        />
        <Text variant="large-medium" as="h2">
          This chat isn&apos;t available on this account
        </Text>
        <Text variant="body" className="text-muted-foreground">
          It may belong to another account you use, or it no longer exists.
        </Text>
        <Button
          type="button"
          variant="secondary"
          size="small"
          className="mt-2"
          onClick={handleNewChat}
        >
          Start a new chat
        </Button>
      </div>
    </div>
  );
}
