"use client";

import { ChatInterface } from "@/components/chat/ChatInterface";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import { useEffect } from "react";
import { useSearchParams } from "next/navigation";
import BackendAPI from "@/lib/autogpt-server-api";

export default function DiscoverPage() {
  const { user } = useSupabase();
  const searchParams = useSearchParams();
  const sessionId = searchParams.get("sessionId");

  useEffect(() => {
    // Check if we need to assign user to anonymous session after login
    const assignSessionToUser = async () => {
      // Priority 1: Session from URL (user returning from auth)
      const urlSession = sessionId;
      // Priority 2: Pending session from localStorage
      const pendingSession = localStorage.getItem("pending_chat_session");

      const sessionToAssign = urlSession || pendingSession;

      if (sessionToAssign && user) {
        try {
          const api = new BackendAPI();
          // Call the assign-user endpoint
          await (api as any)._request(
            "PATCH",
            `/v2/chat/sessions/${sessionToAssign}/assign-user`,
            {},
          );

          // Clear the pending session flag
          localStorage.removeItem("pending_chat_session");

          // The session is now owned by the user
          console.log(
            `Session ${sessionToAssign} assigned to user successfully`,
          );
        } catch (e: any) {
          // Check if error is because session already has a user
          if (e.message?.includes("already has an assigned user")) {
            console.log("Session already assigned to user, continuing...");
          } else {
            console.error("Failed to assign session to user:", e);
          }
        }
      }
    };

    if (user) {
      assignSessionToUser();
    }
  }, [user, sessionId]);

  return (
    <div className="h-screen">
      <ChatInterface
        sessionId={sessionId || undefined}
        systemPrompt="You are a helpful assistant that helps users discover and set up AI agents from the AutoGPT marketplace. Be conversational, friendly, and guide users through finding the right agent for their needs. When users describe what they want to accomplish, search for relevant agents and present them in an engaging way. Help them understand what each agent does and guide them through the setup process."
      />
    </div>
  );
}
