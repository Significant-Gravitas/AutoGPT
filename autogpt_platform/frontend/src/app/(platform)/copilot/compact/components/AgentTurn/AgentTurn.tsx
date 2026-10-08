import type { UIMessage } from "ai";
import { ChainMessageParts } from "../../../components/ChatMessagesContainer/components/ChainMessageParts";
import { MessagePartRenderer } from "../../../components/ChatMessagesContainer/components/MessagePartRenderer";
import type { MessagePart } from "../../../components/ChatMessagesContainer/helpers";
import { buildCompactBlocks } from "../../helpers";
import { ActivityLine } from "../ActivityLine/ActivityLine";

interface Props {
  message: UIMessage;
  isStreaming: boolean;
}

export function AgentTurn({ message, isStreaming }: Props) {
  const blocks = buildCompactBlocks(message.parts as MessagePart[]);
  const lastActivity = blocks.findLastIndex(
    (block) => block.kind === "activity",
  );

  return blocks.map((block, blockIndex) => {
    const key = `${message.id}-${block.kind}-${block.index}`;
    if (block.kind === "activity") {
      return (
        <ActivityLine
          key={key}
          parts={block.parts}
          isStreaming={isStreaming && blockIndex === lastActivity}
        />
      );
    }
    if (block.kind === "experts") {
      return (
        <ChainMessageParts
          key={key}
          parts={block.parts}
          messageID={message.id}
          isCurrentlyStreaming={isStreaming}
        />
      );
    }
    return (
      <MessagePartRenderer
        key={key}
        part={block.part}
        messageID={message.id}
        partIndex={block.index}
        isCurrentlyStreaming={isStreaming}
      />
    );
  });
}
