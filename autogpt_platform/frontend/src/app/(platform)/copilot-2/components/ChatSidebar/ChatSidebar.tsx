import { useState } from "react";
import { postV2CreateSession } from "@/app/api/__generated__/endpoints/chat/chat";
import { parseAsString, useQueryState } from "nuqs";

export const ChatSidebar = ({
  isCreating,
  setIsCreating,
}: {
  isCreating: boolean;
  setIsCreating: (isCreating: boolean) => void;
}) => {
  const [sessionId, setSessionId] = useQueryState("sessionId", parseAsString);

  async function createSession(): Promise<string | null> {
    setIsCreating(true);
    try {
      const response = await postV2CreateSession({
        body: JSON.stringify({}),
      });
      if (response.status === 200 && response.data?.id) {
        return response.data.id;
      }
      return null;
    } catch (error) {
      return null;
    } finally {
      setIsCreating(false);
    }
  }

  async function handleNewSession() {
    const newSessionId = await createSession();
    if (newSessionId) {
      setSessionId(newSessionId);
    }
  }

  return (
    <div className="flex w-64 flex-col border-r border-zinc-200 bg-zinc-50 p-4">
      <button
        onClick={handleNewSession}
        disabled={isCreating}
        className="rounded-md bg-blue-600 px-4 py-2 text-white transition-colors hover:bg-blue-700 disabled:opacity-50"
      >
        {isCreating ? "Creating..." : "New Session"}
      </button>
    </div>
  );
};
