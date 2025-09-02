"use client";

import React, { useState, useEffect, useRef } from "react";
import { ChatMessage } from "@/components/chat/ChatMessage";
import { ChatInput } from "@/components/chat/ChatInput";
import { ToolCallWidget } from "@/components/chat/ToolCallWidget";
import { AgentDiscoveryCard } from "@/components/chat/AgentDiscoveryCard";
import { ChatMessage as ChatMessageType } from "@/lib/autogpt-server-api/chat";
import { Loader2 } from "lucide-react";

// Demo page that simulates the chat interface without requiring authentication
export default function DiscoverDemoPage() {
  const [messages, setMessages] = useState<ChatMessageType[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamingContent, setStreamingContent] = useState("");
  const [toolCalls, setToolCalls] = useState<any[]>([]);
  const [discoveredAgents, setDiscoveredAgents] = useState<any[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, streamingContent, toolCalls]);

  // Add welcome message on mount
  useEffect(() => {
    setMessages([
      {
        content: "Hello! I'm your AI agent discovery assistant. I can help you find and set up the perfect AI agents for your needs. What would you like to automate today?",
        role: "ASSISTANT",
        created_at: new Date().toISOString(),
      },
    ]);
  }, []);

  const simulateResponse = async (message: string) => {
    setIsStreaming(true);
    setStreamingContent("");
    setToolCalls([]);
    
    // Add user message
    const userMessage: ChatMessageType = {
      content: message,
      role: "USER",
      created_at: new Date().toISOString(),
    };
    setMessages((prev) => [...prev, userMessage]);

    // Simulate thinking
    await new Promise((resolve) => setTimeout(resolve, 500));

    // Check for keywords and simulate appropriate response
    const lowerMessage = message.toLowerCase();
    
    if (lowerMessage.includes("content") || lowerMessage.includes("write") || lowerMessage.includes("blog")) {
      // Simulate tool call
      setToolCalls([{
        id: "tool-1",
        name: "find_agent",
        parameters: { search_query: "content creation" },
        status: "calling",
      }]);
      
      await new Promise((resolve) => setTimeout(resolve, 1000));
      
      setToolCalls([{
        id: "tool-1",
        name: "find_agent",
        parameters: { search_query: "content creation" },
        status: "executing",
      }]);
      
      await new Promise((resolve) => setTimeout(resolve, 1500));
      
      setToolCalls([{
        id: "tool-1",
        name: "find_agent",
        parameters: { search_query: "content creation" },
        status: "completed",
        result: "Found 3 agents for content creation",
      }]);
      
      // Simulate discovered agents
      setDiscoveredAgents([
        {
          id: "agent-001",
          version: "1.0.0",
          name: "Blog Writer Pro",
          description: "Generates high-quality blog posts with SEO optimization",
          creator: "AutoGPT Team",
          rating: 4.8,
          runs: 5420,
          categories: ["Content", "SEO", "Marketing"],
        },
        {
          id: "agent-002",
          version: "2.1.0",
          name: "Social Media Content Creator",
          description: "Creates engaging social media posts for multiple platforms",
          creator: "Community",
          rating: 4.6,
          runs: 3200,
          categories: ["Social Media", "Marketing"],
        },
        {
          id: "agent-003",
          version: "1.5.0",
          name: "Technical Documentation Writer",
          description: "Generates comprehensive technical documentation from code",
          creator: "DevTools Inc",
          rating: 4.9,
          runs: 2100,
          categories: ["Documentation", "Development"],
        },
      ]);
      
      // Simulate streaming response
      const response = "I found some excellent content creation agents for you! These agents can help with blog writing, social media content, and technical documentation. Each one has been highly rated by the community.";
      
      for (let i = 0; i < response.length; i += 5) {
        setStreamingContent(response.substring(0, i + 5));
        await new Promise((resolve) => setTimeout(resolve, 50));
      }
      
      setMessages((prev) => [...prev, {
        content: response,
        role: "ASSISTANT",
        created_at: new Date().toISOString(),
      }]);
      
    } else if (lowerMessage.includes("automat") || lowerMessage.includes("task")) {
      // Different response for automation
      const response = "I can help you find automation agents! What specific tasks would you like to automate? For example:\n\n- Data processing and analysis\n- Email management\n- File organization\n- Web scraping\n- Report generation\n- API integrations\n\nJust describe what you need and I'll find the perfect agent for you!";
      
      for (let i = 0; i < response.length; i += 5) {
        setStreamingContent(response.substring(0, i + 5));
        await new Promise((resolve) => setTimeout(resolve, 30));
      }
      
      setMessages((prev) => [...prev, {
        content: response,
        role: "ASSISTANT",
        created_at: new Date().toISOString(),
      }]);
      
    } else {
      // Generic response
      const response = `I understand you're interested in "${message}". Let me search for relevant agents that can help you with that.`;
      
      for (let i = 0; i < response.length; i += 5) {
        setStreamingContent(response.substring(0, i + 5));
        await new Promise((resolve) => setTimeout(resolve, 40));
      }
      
      setMessages((prev) => [...prev, {
        content: response,
        role: "ASSISTANT",
        created_at: new Date().toISOString(),
      }]);
    }
    
    setStreamingContent("");
    setIsStreaming(false);
  };

  const handleSendMessage = (message: string) => {
    if (!isStreaming) {
      simulateResponse(message);
    }
  };

  const handleSelectAgent = (agent: any) => {
    handleSendMessage(`I want to set up the agent "${agent.name}"`);
  };

  const handleGetAgentDetails = (agent: any) => {
    handleSendMessage(`Tell me more about "${agent.name}"`);
  };

  return (
    <div className="flex h-screen flex-col bg-neutral-50 dark:bg-neutral-950">
      {/* Header */}
      <div className="border-b border-neutral-200 dark:border-neutral-700 bg-white dark:bg-neutral-900 px-4 py-3">
        <div className="mx-auto max-w-4xl">
          <h1 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">
            AI Agent Discovery Assistant (Demo)
          </h1>
          <p className="text-sm text-neutral-600 dark:text-neutral-400">
            This is a demo of the chat-based agent discovery interface
          </p>
        </div>
      </div>

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto">
        <div className="mx-auto max-w-4xl py-4">
          {messages.map((message, index) => (
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

          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input Area */}
      <ChatInput
        onSendMessage={handleSendMessage}
        isStreaming={isStreaming}
        placeholder="Try: 'I need help with content creation' or 'Show me automation agents'"
      />
    </div>
  );
}