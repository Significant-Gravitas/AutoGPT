"use client";

import React from "react";
import { ChatMessage as ChatMessageType } from "@/lib/autogpt-server-api/chat";
import { cn } from "@/lib/utils";
import { User, Bot } from "lucide-react";

interface ChatMessageProps {
  message: ChatMessageType;
  className?: string;
}

export function ChatMessage({ message, className }: ChatMessageProps) {
  const isUser = message.role === "USER";
  const isAssistant = message.role === "ASSISTANT";
  const isSystem = message.role === "SYSTEM";
  const isTool = message.role === "TOOL";

  return (
    <div
      className={cn(
        "flex gap-4 px-4 py-6",
        isUser && "justify-end",
        !isUser && "justify-start",
        className
      )}
    >
      {!isUser && (
        <div className="flex-shrink-0">
          <div className="flex h-8 w-8 items-center justify-center rounded-full bg-violet-600">
            <Bot className="h-5 w-5 text-white" />
          </div>
        </div>
      )}
      
      <div
        className={cn(
          "max-w-[70%] rounded-lg px-4 py-3",
          isUser && "bg-neutral-100 dark:bg-neutral-800",
          isAssistant && "bg-white dark:bg-neutral-900 border border-neutral-200 dark:border-neutral-700",
          isSystem && "bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800",
          isTool && "bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800"
        )}
      >
        {isSystem && (
          <div className="mb-2 text-xs font-medium text-blue-600 dark:text-blue-400">
            System
          </div>
        )}
        
        {isTool && (
          <div className="mb-2 text-xs font-medium text-green-600 dark:text-green-400">
            Tool Response
          </div>
        )}
        
        <div className="prose prose-sm dark:prose-invert max-w-none">
          {/* Simple markdown-like rendering without external dependencies */}
          <div className="whitespace-pre-wrap">
            {message.content.split('\n').map((line, index) => {
              // Basic markdown parsing
              if (line.startsWith('# ')) {
                return <h1 key={index} className="text-xl font-bold mb-2">{line.substring(2)}</h1>;
              } else if (line.startsWith('## ')) {
                return <h2 key={index} className="text-lg font-bold mb-2">{line.substring(3)}</h2>;
              } else if (line.startsWith('### ')) {
                return <h3 key={index} className="text-base font-bold mb-2">{line.substring(4)}</h3>;
              } else if (line.startsWith('- ')) {
                return <li key={index} className="list-disc ml-4">{line.substring(2)}</li>;
              } else if (line.startsWith('```')) {
                return <pre key={index} className="bg-neutral-100 dark:bg-neutral-800 p-2 rounded my-2 overflow-x-auto"><code>{line.substring(3)}</code></pre>;
              } else if (line.trim() === '') {
                return <br key={index} />;
              } else {
                return <p key={index} className="mb-1">{line}</p>;
              }
            })}
          </div>
        </div>
        
        {message.tokens && (
          <div className="mt-3 flex gap-3 text-xs text-neutral-500 dark:text-neutral-400">
            {message.tokens.prompt && (
              <span>Prompt: {message.tokens.prompt}</span>
            )}
            {message.tokens.completion && (
              <span>Completion: {message.tokens.completion}</span>
            )}
            {message.tokens.total && (
              <span>Total: {message.tokens.total}</span>
            )}
          </div>
        )}
      </div>
      
      {isUser && (
        <div className="flex-shrink-0">
          <div className="flex h-8 w-8 items-center justify-center rounded-full bg-neutral-600">
            <User className="h-5 w-5 text-white" />
          </div>
        </div>
      )}
    </div>
  );
}