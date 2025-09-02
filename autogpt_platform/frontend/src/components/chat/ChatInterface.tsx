"use client";

import React, { useState, useEffect, useRef, useCallback } from "react";
import { ChatMessage } from "./ChatMessage";
import { ChatInput } from "./ChatInput";
import { ToolCallWidget } from "./ToolCallWidget";
import { AgentDiscoveryCard } from "./AgentDiscoveryCard";
import { AuthPromptWidget } from "./AuthPromptWidget";
import { useChatSession } from "@/hooks/useChatSession";
import { useChatStream } from "@/hooks/useChatStream";
import { ChatMessage as ChatMessageType, StreamChunk } from "@/lib/autogpt-server-api/chat";
import { cn } from "@/lib/utils";
import { Loader2 } from "lucide-react";

interface ToolCall {
  id: string;
  name: string;
  parameters: Record<string, any>;
  status: "calling" | "executing" | "completed" | "error";
  result?: string;
  error?: string;
}

interface ChatInterfaceProps {
  className?: string;
  systemPrompt?: string;
}

export function ChatInterface({ className, systemPrompt }: ChatInterfaceProps) {
  const { session, messages, isLoading, error, createSession } = useChatSession();
  const { isStreaming, sendMessage, stopStreaming } = useChatStream();
  
  const [localMessages, setLocalMessages] = useState<ChatMessageType[]>([]);
  const [streamingContent, setStreamingContent] = useState("");
  const [toolCalls, setToolCalls] = useState<ToolCall[]>([]);
  const [discoveredAgents, setDiscoveredAgents] = useState<any[]>([]);
  const [authPrompt, setAuthPrompt] = useState<any>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Initialize session on mount
  useEffect(() => {
    if (!session) {
      createSession(systemPrompt);
    }
    
    // Check for pending agent setup after authentication
    const pendingAgentSetup = localStorage.getItem("pending_agent_setup");
    if (pendingAgentSetup && session) {
      try {
        const agentInfo = JSON.parse(pendingAgentSetup);
        // Clear the pending setup
        localStorage.removeItem("pending_agent_setup");
        // Trigger agent setup
        handleSendMessage(`Set up the agent "${agentInfo.name}" (ID: ${agentInfo.graph_id})`);
      } catch (e) {
        console.error("Failed to parse pending agent setup:", e);
      }
    }
  }, [session]);

  // Sync messages
  useEffect(() => {
    setLocalMessages(messages);
  }, [messages]);

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [localMessages, streamingContent, toolCalls]);

  const handleSendMessage = useCallback(async (message: string) => {
    if (!session) {
      await createSession(systemPrompt);
      return;
    }

    // Add user message immediately
    const userMessage: ChatMessageType = {
      content: message,
      role: "USER",
      created_at: new Date().toISOString(),
    };
    setLocalMessages((prev) => [...prev, userMessage]);

    // Clear streaming content
    setStreamingContent("");
    setToolCalls([]);

    // Start streaming
    let assistantContent = "";
    let currentToolCall: ToolCall | null = null;

    await sendMessage(session.id, message, (chunk: StreamChunk) => {
      if (chunk.type === "text") {
        assistantContent += chunk.content;
        setStreamingContent(assistantContent);
      } else if (chunk.type === "html") {
        // Parse HTML content for tool calls and special widgets
        handleHtmlContent(chunk.content);
      } else if (chunk.type === "error") {
        console.error("Stream error:", chunk.content);
      }
    });

    // Add final assistant message
    if (assistantContent) {
      const assistantMessage: ChatMessageType = {
        content: assistantContent,
        role: "ASSISTANT",
        created_at: new Date().toISOString(),
      };
      setLocalMessages((prev) => [...prev, assistantMessage]);
      setStreamingContent("");
    }
  }, [session, sendMessage, createSession, systemPrompt]);

  const handleHtmlContent = (html: string) => {
    // Parse the HTML to detect tool calls and extract data
    const parser = new DOMParser();
    const doc = parser.parseFromString(html, "text/html");
    
    // Check for tool call containers
    const toolCallContainer = doc.querySelector(".tool-call-container");
    if (toolCallContainer) {
      const toolHeader = toolCallContainer.querySelector(".tool-header");
      const toolBody = toolCallContainer.querySelector(".tool-body");
      
      if (toolHeader) {
        const toolNameMatch = toolHeader.textContent?.match(/Calling Tool: (\w+)/);
        if (toolNameMatch) {
          const toolName = toolNameMatch[1];
          const preElement = toolBody?.querySelector("pre");
          let parameters = {};
          
          try {
            if (preElement?.textContent) {
              parameters = JSON.parse(preElement.textContent);
            }
          } catch (e) {
            console.error("Failed to parse tool parameters:", e);
          }
          
          const toolCall: ToolCall = {
            id: `tool-${Date.now()}`,
            name: toolName,
            parameters,
            status: "calling",
          };
          
          setToolCalls((prev) => [...prev, toolCall]);
        }
      }
    }
    
    // Check for executing indicator
    const executingDiv = doc.querySelector(".tool-executing");
    if (executingDiv) {
      setToolCalls((prev) => {
        const updated = [...prev];
        if (updated.length > 0) {
          updated[updated.length - 1].status = "executing";
        }
        return updated;
      });
    }
    
    // Check for tool result
    const resultDiv = doc.querySelector(".tool-result");
    if (resultDiv) {
      const resultContent = resultDiv.querySelector("div:last-child")?.textContent || "";
      
      // Check for auth_required response
      if (resultContent.includes('"status": "auth_required"')) {
        try {
          const jsonMatch = resultContent.match(/\{[\s\S]*\}/);
          if (jsonMatch) {
            const authData = JSON.parse(jsonMatch[0]);
            if (authData.status === "auth_required") {
              setAuthPrompt({
                message: authData.message,
                sessionId: authData.session_id || session?.id,
                agentInfo: authData.agent_info,
              });
            }
          }
        } catch (e) {
          console.error("Failed to parse auth response:", e);
        }
      }
      // Parse agent search results
      else if (resultContent.includes("agents matching")) {
        try {
          const jsonMatch = resultContent.match(/\[[\s\S]*\]/);
          if (jsonMatch) {
            const agents = JSON.parse(jsonMatch[0]);
            setDiscoveredAgents(agents);
          }
        } catch (e) {
          console.error("Failed to parse agent results:", e);
        }
      }
      
      setToolCalls((prev) => {
        const updated = [...prev];
        if (updated.length > 0) {
          updated[updated.length - 1].status = "completed";
          updated[updated.length - 1].result = resultContent;
        }
        return updated;
      });
    }
  };

  const handleSelectAgent = (agent: any) => {
    // Send a message to set up the selected agent
    handleSendMessage(`I want to set up the agent "${agent.name}" (ID: ${agent.id})`);
  };

  const handleGetAgentDetails = (agent: any) => {
    // Send a message to get more details about the agent
    handleSendMessage(`Tell me more about the agent "${agent.name}" (ID: ${agent.id})`);
  };

  if (isLoading && !session) {
    return (
      <div className="flex h-screen items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-violet-600" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex h-screen items-center justify-center">
        <div className="text-center">
          <p className="text-red-500">Error: {error.message}</p>
          <button
            onClick={() => window.location.reload()}
            className="mt-4 rounded-lg bg-violet-600 px-4 py-2 text-white hover:bg-violet-700"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className={cn("flex h-screen flex-col bg-neutral-50 dark:bg-neutral-950", className)}>
      {/* Header */}
      <div className="border-b border-neutral-200 dark:border-neutral-700 bg-white dark:bg-neutral-900 px-4 py-3">
        <div className="mx-auto max-w-4xl">
          <h1 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">
            AI Agent Discovery Assistant
          </h1>
          <p className="text-sm text-neutral-600 dark:text-neutral-400">
            Chat with me to find and set up the perfect AI agent for your needs
          </p>
        </div>
      </div>

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto">
        <div className="mx-auto max-w-4xl py-4">
          {localMessages.length === 0 && !streamingContent && (
            <div className="px-4 py-8 text-center">
              <p className="text-neutral-600 dark:text-neutral-400">
                👋 Hello! I'm here to help you discover and set up AI agents.
              </p>
              <p className="mt-2 text-sm text-neutral-500 dark:text-neutral-500">
                Try asking: "I need help with content creation" or "Show me automation agents"
              </p>
            </div>
          )}

          {localMessages.map((message, index) => (
            <ChatMessage key={index} message={message} />
          ))}

          {streamingContent && (
            <ChatMessage
              message={{
                content: streamingContent,
                role: "ASSISTANT",
              }}
            />
          )}

          {toolCalls.map((toolCall) => (
            <div key={toolCall.id} className="px-4">
              <ToolCallWidget
                toolName={toolCall.name}
                parameters={toolCall.parameters}
                result={toolCall.result}
                status={toolCall.status}
                error={toolCall.error}
              />
            </div>
          ))}

          {discoveredAgents.length > 0 && (
            <div className="px-4">
              <AgentDiscoveryCard
                agents={discoveredAgents}
                onSelectAgent={handleSelectAgent}
                onGetDetails={handleGetAgentDetails}
              />
            </div>
          )}

          {authPrompt && (
            <div className="px-4">
              <AuthPromptWidget
                message={authPrompt.message}
                sessionId={authPrompt.sessionId}
                agentInfo={authPrompt.agentInfo}
              />
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input Area */}
      <ChatInput
        onSendMessage={handleSendMessage}
        onStopStreaming={stopStreaming}
        isStreaming={isStreaming}
        disabled={!session}
      />
    </div>
  );
}