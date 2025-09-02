"use client";

import { ChatInterface } from "@/components/chat/ChatInterface";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import { useEffect } from "react";
import BackendAPI from "@/lib/autogpt-server-api";

export default function DiscoverPage() {
  const { user } = useSupabase();
  
  useEffect(() => {
    // Check if we need to assign user to anonymous session after login
    const assignSessionToUser = async () => {
      const pendingSession = localStorage.getItem("pending_chat_session");
      if (pendingSession && user) {
        try {
          const api = new BackendAPI();
          // Call the assign-user endpoint
          await (api as any)._request(
            "PATCH",
            `/v2/chat/sessions/${pendingSession}/assign-user`,
            {}
          );
          
          // Clear the pending session flag
          localStorage.removeItem("pending_chat_session");
          
          // The session is now owned by the user
          console.log("Session assigned to user successfully");
        } catch (e) {
          console.error("Failed to assign session to user:", e);
        }
      }
    };
    
    if (user) {
      assignSessionToUser();
    }
  }, [user]);
  
  return (
    <div className="h-screen">
      <ChatInterface 
        systemPrompt="You are a helpful assistant that helps users discover and set up AI agents from the AutoGPT marketplace. Be conversational, friendly, and guide users through finding the right agent for their needs. When users describe what they want to accomplish, search for relevant agents and present them in an engaging way. Help them understand what each agent does and guide them through the setup process."
      />
    </div>
  );
}