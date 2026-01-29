'use client';

import { useChat } from '@ai-sdk/react';
import { DefaultChatTransport } from 'ai';
import { useState, useMemo } from 'react';
import { parseAsString, useQueryState } from 'nuqs';
import { postV2CreateSession } from '@/app/api/__generated__/endpoints/chat/chat';

export default function Page() {
  const [sessionId, setSessionId] = useQueryState('sessionId', parseAsString);
  const [isCreating, setIsCreating] = useState(false);
  const [input, setInput] = useState('');

  const transport = useMemo(() => {
    if (!sessionId) return null;
    return new DefaultChatTransport({
      api: `/api/chat/sessions/${sessionId}/stream`,
      prepareSendMessagesRequest: ({ id, messages, trigger, messageId }) => {
        const last = messages[messages.length - 1];
        return {
          body: {
            message: last.parts?.map((p) => (p.type === 'text' ? p.text : '')).join(''),
            is_user_message: last.role === 'user',
            context: null,
          },
        };
      },
    });
  }, [sessionId]);

  const { messages, sendMessage, status, error } = useChat({
    transport: transport ?? undefined,
  });

  async function createSession(): Promise<string | null> {
    setIsCreating(true);
    try {
      const response = await postV2CreateSession({
        body: JSON.stringify({}),
      });
      if (response.status === 200 && response.data?.id) {
        return response.data.id;
      }
      console.error('[Copilot2] Failed to create session:', response);
      return null;
    } catch (error) {
      console.error('[Copilot2] Error creating session:', error);
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

  async function handleStartChat(prompt: string) {
    if (!prompt.trim() || isCreating) return;
    
    const newSessionId = await createSession();
    if (newSessionId) {
      await setSessionId(newSessionId);
      sendMessage({ text: prompt });
      setInput('');
    }
  }

  // Show landing page when no session exists
  if (!sessionId) {
    return (
      <div className="flex h-full">
        <div className="flex w-64 flex-col border-r border-zinc-200 bg-zinc-50 p-4">
          <button
            onClick={handleNewSession}
            disabled={isCreating}
            className="rounded-md bg-blue-600 px-4 py-2 text-white transition-colors hover:bg-blue-700 disabled:opacity-50"
          >
            {isCreating ? 'Creating...' : 'New Session'}
          </button>
        </div>

        <div className="flex h-full flex-1 flex-col items-center justify-center bg-zinc-100 p-4">
          <h2 className="mb-4 text-xl font-semibold text-zinc-700">Start a new conversation</h2>
          <form
            onSubmit={(e) => {
              e.preventDefault();
              handleStartChat(input);
            }}
            className="w-full max-w-md"
          >
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              disabled={isCreating}
              placeholder="Type your message to start..."
              className="w-full rounded-md border border-zinc-300 px-4 py-2"
            />
            <button
              type="submit"
              disabled={isCreating || !input.trim()}
              className="mt-2 w-full rounded-md bg-blue-600 px-4 py-2 text-white transition-colors hover:bg-blue-700 disabled:opacity-50"
            >
              {isCreating ? 'Starting...' : 'Start Chat'}
            </button>
          </form>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-full">
      {/* Sidebar */}
      <div className="flex w-64 flex-col border-r border-zinc-200 bg-zinc-50 p-4">
        <button
          onClick={handleNewSession}
          disabled={isCreating}
          className="rounded-md bg-blue-600 px-4 py-2 text-white transition-colors hover:bg-blue-700 disabled:opacity-50"
        >
          {isCreating ? 'Creating...' : 'New Session'}
        </button>
      </div>

      {/* Chat area */}
      <div className="flex h-full flex-1 flex-col bg-zinc-100 p-4">
        <div className="text-sm text-zinc-500 mb-2">Session ID: {sessionId}</div>
        <div className="flex-1 overflow-y-auto">
          {messages.map((message) => (
            <div key={message.id}>
              {message.role === 'user' ? 'User: ' : 'AI: '}
              {message.parts.map((part, index) =>
                part.type === 'text' ? <p key={index}>{part.text}</p> : null,
              )}
            </div>
          ))}
          {error && <div className="text-red-500">Error: {error.message}</div>}
        </div>

        <form
          onSubmit={(e) => {
            e.preventDefault();
            if (input.trim()) {
              sendMessage({ text: input });
              setInput('');
            }
          }}
        >
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            disabled={status !== 'ready'}
            placeholder="Say something..."
          />
          <button type="submit" disabled={status !== 'ready'}>
            Submit
          </button>
        </form>
      </div>
    </div>
  );
}