import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { BubbleChatIcon } from "@hugeicons/core-free-icons";
import { getChatWithExpertHref } from "../helpers";

interface Props {
  expertId: string;
  expertName: string;
  onDone: () => void;
}

// A reset expert has no chats left, so it drops out of the sidebar: this is
// where the user finds the way back to it.
export function ResetDone({ expertId, expertName, onDone }: Props) {
  return (
    <div className="flex flex-col gap-4">
      <Text variant="body" tone="secondary">
        {expertName} starts over. Their memory and computer are erased, and
        your old chats with them are removed.
      </Text>
      <Text variant="body" tone="secondary">
        Chat with {expertName} and they&apos;ll introduce themselves again.
      </Text>
      <Dialog.Footer>
        <Button size="small" variant="secondary" onClick={onDone}>
          Done
        </Button>
        <Button
          as="NextLink"
          href={getChatWithExpertHref(expertId)}
          size="small"
          variant="primary"
          leadingIcon={BubbleChatIcon}
          data-testid="reset-expert-chat"
        >
          Chat with {expertName}
        </Button>
      </Dialog.Footer>
    </div>
  );
}
