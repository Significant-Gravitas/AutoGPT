import { Text } from "@/components/atoms/Text/Text";
import { ChatLoader } from "@/components/contextual/Chat/components/ChatLoader/ChatLoader";

export function LoadingState() {
  return (
    <div className="flex flex-1 items-center justify-center">
      <div className="flex flex-col items-center gap-4">
        <ChatLoader />
        <Text variant="body" className="text-zinc-500">
          Loading your chats...
        </Text>
      </div>
    </div>
  );
}
