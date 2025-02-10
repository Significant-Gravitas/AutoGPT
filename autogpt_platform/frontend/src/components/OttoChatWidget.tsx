"use client";

import React, { useEffect, useState, useRef } from 'react';
import { useSearchParams } from 'next/navigation';
import { useToast } from '@/components/ui/use-toast';
import useSupabase from '../hooks/useSupabase';
import useAgentGraph from '../hooks/useAgentGraph';
import ReactMarkdown, { Components } from 'react-markdown';

interface Document {
  url: string;
  relevance_score: number;
}

interface ApiResponse {
  answer: string;
  documents: Document[];
  success: boolean;
}

interface Message {
  type: 'user' | 'assistant';
  content: string;
}

const OttoChatWidget = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [includeGraphData, setIncludeGraphData] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const { user, supabase } = useSupabase();
  const searchParams = useSearchParams();
  const flowID = searchParams.get('flowID');
  const { nodes, edges } = useAgentGraph(flowID || undefined);
  const { toast } = useToast();

  useEffect(() => {
    // Add welcome message when component mounts
    if (messages.length === 0) {
      setMessages([
        { type: 'assistant', content: 'Hello im Otto! Ask me anything about AutoGPT!' }
      ]);
    }
  }, []);

  useEffect(() => {
    // Scroll to bottom whenever messages change
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputValue.trim() || isProcessing) return;

    const userMessage = inputValue.trim();
    setInputValue('');
    setIsProcessing(true);

    // Add user message to chat
    setMessages(prev => [...prev, { type: 'user', content: userMessage }]);

    try {
      if (!supabase) {
        throw new Error('Supabase client not initialized');
      }

      // Get the current session
      const { data: { session } } = await supabase.auth.getSession();
      if (!session) {
        throw new Error('No active session');
      }

      const messageId = `${Date.now()}-web`;

      const payload = {
        query: userMessage,
        conversation_history: messages
          .reduce<{ query: string; response: string; }[]>((acc, msg, i, arr) => {
            if (msg.type === 'user' && i + 1 < arr.length && arr[i + 1].type === 'assistant') {
              acc.push({
                query: msg.content,
                response: arr[i + 1].content
              });
            }
            return acc;
          }, []),
        user_id: user?.id || 'anonymous',
        message_id: messageId,
        include_graph_data: includeGraphData,
        graph_id: flowID || undefined
      };

      setIncludeGraphData(false);

      // Add temporary processing message
      setMessages(prev => [...prev, { type: 'assistant', content: 'Processing your question...' }]);

      const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8006';
      const response = await fetch(`${BACKEND_URL}/api/otto/ask`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
          'Authorization': `Bearer ${session.access_token}`,
        },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        if (response.status === 401) {
          toast({
            title: "Authentication Error",
            description: "Please sign in to use the chat feature.",
            variant: "destructive",
          });
          throw new Error('Authentication required');
        }
        throw new Error(`API request failed with status ${response.status}`);
      }

      const data: ApiResponse = await response.json();
      
      // Remove processing message and add actual response
      setMessages(prev => [...prev.slice(0, -1), { type: 'assistant', content: data.answer }]);

    } catch (error) {
      console.error('Error calling API:', error);
      // Remove processing message and add error message
      const errorMessage = error instanceof Error && error.message === 'No active session'
        ? 'Please sign in to use the chat feature.'
        : 'Sorry, there was an error processing your message. Please try again.';
      
      setMessages(prev => [
        ...prev.slice(0, -1), 
        { type: 'assistant', content: errorMessage }
      ]);
    } finally {
      setIsProcessing(false);
    }
  };

  if (!isOpen) {
    return (
      <div className="fixed bottom-4 right-4 z-50">
        <button 
          onClick={() => setIsOpen(true)}
          className="bg-primary text-primary-foreground rounded-full p-3 shadow-lg hover:bg-primary/90 transition-colors"
          aria-label="Open chat widget"
        >
          <svg 
            viewBox="0 0 24 24" 
            className="w-6 h-6"
            stroke="currentColor" 
            strokeWidth="2" 
            fill="none" 
            strokeLinecap="round" 
            strokeLinejoin="round"
          >
            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
          </svg>
        </button>
      </div>
    );
  }

  return (
    <div className="fixed bottom-4 right-4 w-[600px] h-[600px] bg-background border rounded-lg shadow-xl flex flex-col z-50">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b">
        <h2 className="font-semibold">Otto Assistant</h2>
        <button 
          onClick={() => setIsOpen(false)}
          className="text-muted-foreground hover:text-foreground transition-colors"
          aria-label="Close chat"
        >
          <svg 
            viewBox="0 0 24 24" 
            className="w-5 h-5"
            stroke="currentColor" 
            strokeWidth="2" 
            fill="none" 
            strokeLinecap="round" 
            strokeLinejoin="round"
          >
            <line x1="18" y1="6" x2="6" y2="18" />
            <line x1="6" y1="6" x2="18" y2="18" />
          </svg>
        </button>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((message, index) => (
          <div
            key={index}
            className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-[80%] rounded-lg p-3 ${
                message.type === 'user'
                  ? 'bg-black text-white ml-4'
                  : 'bg-[#8b5cf6] text-white mr-4'
              }`}
            >
              {message.type === 'user' ? (
                message.content
              ) : (
                <ReactMarkdown
                  className="prose prose-sm dark:prose-invert max-w-none"
                  components={{
                    p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,
                    code(props) {
                      const { children, className, node, ...rest } = props
                      const match = /language-(\w+)/.exec(className || '')
                      return match ? (
                        <pre className="p-3 rounded-md bg-muted-foreground/20 overflow-x-auto">
                          <code className="text-sm font-mono" {...rest}>
                            {children}
                          </code>
                        </pre>
                      ) : (
                        <code className="px-1 py-0.5 rounded-md bg-muted-foreground/20 font-mono text-sm" {...rest}>
                          {children}
                        </code>
                      )
                    },
                    ul: ({ children }) => <ul className="list-disc pl-4 mb-2 last:mb-0">{children}</ul>,
                    ol: ({ children }) => <ol className="list-decimal pl-4 mb-2 last:mb-0">{children}</ol>,
                    li: ({ children }) => <li className="mb-1 last:mb-0">{children}</li>,
                  }}
                >
                  {message.content}
                </ReactMarkdown>
              )}
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <form onSubmit={handleSubmit} className="p-4 border-t">
        <div className="flex flex-col gap-2">
          <div className="flex gap-2">
            <input
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              placeholder="Type your message..."
              className="flex-1 bg-background border rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-primary"
              disabled={isProcessing}
            />
            <button
              type="submit"
              disabled={isProcessing}
              className="bg-primary text-primary-foreground px-4 py-2 rounded-md hover:bg-primary/90 transition-colors disabled:opacity-50"
            >
              Send
            </button>
          </div>
          {nodes && edges && (
            <button
              type="button"
              onClick={() => {
                setIncludeGraphData(prev => !prev);
              }}
              className={`flex items-center gap-2 text-sm px-2 py-1.5 rounded border transition-all duration-200 ${
                includeGraphData 
                  ? 'bg-primary/10 text-primary border-primary/30 hover:shadow-[0_0_10px_3px_rgba(139,92,246,0.3)]' 
                  : 'bg-muted text-muted-foreground border-transparent hover:bg-muted/80 hover:shadow-[0_0_10px_3px_rgba(139,92,246,0.15)]'
              }`}
            >
              <svg
                viewBox="0 0 24 24"
                className="w-4 h-4"
                stroke="currentColor"
                strokeWidth="2"
                fill="none"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
                <circle cx="8.5" cy="8.5" r="1.5" />
                <polyline points="21 15 16 10 5 21" />
              </svg>
              {includeGraphData ? 'Graph data will be included' : 'Include graph data'}
            </button>
          )}
        </div>
      </form>
    </div>
  );
};

export default OttoChatWidget;