"use client";

import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { useRouter } from "next/navigation";
import { useEffect } from "react";
import { Chat } from "./components/Chat/Chat";

export default function ChatPage() {
  const isChatEnabled = useGetFlag(Flag.CHAT);
  const router = useRouter();

  useEffect(() => {
    if (isChatEnabled === false) {
      router.push("/marketplace");
    }
  }, [isChatEnabled, router]);

  if (isChatEnabled === null || isChatEnabled === false) {
    return null;
  }

  return (
    <div className="flex h-full flex-col">
      <Chat className="flex-1" />
    </div>
  );
}
