"use client";

import React, { useEffect, useState, useRef } from 'react';
import { Widget, addResponseMessage, addLinkSnippet, deleteMessages } from 'react-chat-widget';
import 'react-chat-widget/lib/styles.css';
import './OttoChatWidget.css';
import useSupabase from '../hooks/useSupabase';
import useAgentGraph from '../hooks/useAgentGraph';
import { Node, Edge } from '@xyflow/react';
import { useSearchParams } from 'next/navigation';
import { useToast } from '@/components/ui/use-toast';

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
  query: string;
  response: string;
}

interface GraphData {
  nodes: {
    id: string;
    type: string;
    position: { x: number; y: number };
    data: any;
  }[];
  edges: {
    id: string;
    source: string;
    target: string;
    sourceHandle: string | null;
    targetHandle: string | null;
    data: any;
  }[];
}

interface ChatPayload {
  query: string;
  conversation_history: { query: string; response: string }[];
  user_id: string;
  message_id: string;
  graph_data?: GraphData;
}

const OttoChatWidget = () => {
  const [chatWindowOpen, setChatWindowOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [includeGraphData, setIncludeGraphData] = useState(false);
  const welcomeMessageSent = useRef(false);
  const processingMessageId = useRef<number | null>(null);
  const { user } = useSupabase();
  const searchParams = useSearchParams();
  const flowID = searchParams.get('flowID');
  const { nodes, edges } = useAgentGraph(flowID || undefined);
  const { toast } = useToast();

  useEffect(() => {
    if (!welcomeMessageSent.current) {
      addResponseMessage('Hello im Otto! Ask me anything about AutoGPT!');
      welcomeMessageSent.current = true;
    }
  }, []);

  const formatResponse = (data: ApiResponse): void => {
    const cleanedResponse = data.answer.replace(/####|###|\*|-/g, '');
    addResponseMessage(cleanedResponse);
  };

  const handleNewUserMessage = async (newMessage: string) => {
    
    // Generate a message ID with timestamp and 'web' suffix, this is used to identify the message in the database
    const messageId = `${Date.now()}-web`;
    
    setMessages(prev => [...prev, { query: newMessage, response: '' }]);
    
    addResponseMessage('Processing your question...');
    
    try {
      const graphData: GraphData | undefined = (includeGraphData && nodes && edges) ? {
        nodes: nodes.map(node => ({
          id: node.id,
          type: node.type || 'custom',
          position: { x: node.position.x, y: node.position.y },
          data: node.data
        })),
        edges: edges.map(edge => ({
          id: edge.id,
          source: edge.source,
          target: edge.target,
          sourceHandle: edge.sourceHandle ?? null,
          targetHandle: edge.targetHandle ?? null,
          data: edge.data || {}
        }))
      } : undefined;

      // Reset the includeGraphData flag after using it
      setIncludeGraphData(false);

      const payload: ChatPayload = {
        query: newMessage,
        conversation_history: messages.map(msg => ({ 
          query: msg.query, 
          response: msg.response 
        })),
        user_id: user?.id || 'anonymous',
        message_id: messageId,
        graph_data: graphData
      };

      const response = await fetch('http://192.168.0.39:2344/ask', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        },
        mode: 'cors',
        credentials: 'omit',
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        throw new Error(`API request failed with status ${response.status}`);
      }

      const data: ApiResponse = await response.json();
      
      deleteMessages(1);
      
      if (data.success) {
        formatResponse(data);
        setMessages(prev => {
          const newMessages = [...prev];
          newMessages[newMessages.length - 1].response = data.answer;
          return newMessages;
        });
      } else {
        throw new Error('API request was not successful');
      }

    } catch (error) {
      deleteMessages(1);
      
      console.error('Error calling API:', error);
      addResponseMessage('Sorry, there was an error processing your message. Please try again.');
    }
  };

  const handleToggle = () => {
    setChatWindowOpen(prev => !prev);
  };

  return (
    <Widget
      handleNewUserMessage={handleNewUserMessage}
      title="Otto Assistant"
      subtitle=""
      handleToggle={handleToggle}
      autofocus={true}
      emojis={true}
      launcher={(handleToggle: () => void) => (
        <div className="launcher-container">
          <button 
            onClick={handleToggle}
            className="custom-launcher-button"
            aria-label="Open chat widget"
          >
            <svg 
              viewBox="0 0 24 24" 
              width="24" 
              height="24" 
              stroke="currentColor" 
              strokeWidth="2" 
              fill="none" 
              strokeLinecap="round" 
              strokeLinejoin="round"
            >
              <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
            </svg>
          </button>
          {chatWindowOpen && nodes && edges && (
            <button
              onClick={() => {
                setIncludeGraphData(prev => {
                  const newState = !prev;
                  toast({
                    title: newState 
                      ? "Graph data will be included with your next message" 
                      : "Graph data will not be included with your next message",
                    duration: 2000,
                  });
                  return newState;
                });
              }}
              className={`capture-graph-button ${includeGraphData ? 'active' : ''}`}
              aria-label="Include graph data with next message"
            >
              <svg
                viewBox="0 0 24 24"
                width="20"
                height="20"
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
            </button>
          )}
        </div>
      )}
    />
  );
};

export default OttoChatWidget;