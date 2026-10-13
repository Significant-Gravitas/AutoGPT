import { toast } from "@/components/molecules/Toast/use-toast";

type ChatSender = (text: string) => void | Promise<void>;

const senders = new Map<string, ChatSender>();
const pending = new Map<string, string[]>();

export function chatHref(sessionId: string) {
  return `/copilot?sessionId=${encodeURIComponent(sessionId)}`;
}

/**
 * Sends `text` to the chat `sessionId` through that chat's own send path
 * (so a running turn queues it as usual). When the chat is not on screen the
 * message waits here until its host mounts and registers.
 */
export async function sendToChat(sessionId: string, text: string) {
  const sender = senders.get(sessionId);
  if (sender) {
    await sender(text);
    return;
  }
  pending.set(sessionId, [...(pending.get(sessionId) ?? []), text]);
}

export function registerChatSender(sessionId: string, sender: ChatSender) {
  senders.set(sessionId, sender);
  const queued = pending.get(sessionId) ?? [];
  pending.delete(sessionId);
  for (const text of queued)
    Promise.resolve(sender(text)).catch(() =>
      toast({
        title: "Couldn't send message",
        description: "Try submitting the view again.",
        variant: "destructive",
      }),
    );
  return () => {
    if (senders.get(sessionId) === sender) senders.delete(sessionId);
  };
}
