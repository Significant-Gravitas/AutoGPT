import { useRef } from "react";
import { useMountEffect } from "@/hooks/useMountEffect";
import { registerChatSender } from "../../helpers/sendToChat";

interface Props {
  sessionId: string;
  onSend: (message: string) => void | Promise<void>;
}

/** Lets views outside the chat column (side panel, Files page) send to this
 *  chat. Mounted once the chat has loaded. */
export function ChatSenderRegistration({ sessionId, onSend }: Props) {
  const latest = useRef(onSend);
  latest.current = onSend;
  useMountEffect(() =>
    registerChatSender(sessionId, (text) => latest.current(text)),
  );
  return null;
}
