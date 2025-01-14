"use client";

import React, { useEffect, useState, useRef } from 'react';
import { Widget, addResponseMessage, addLinkSnippet, deleteMessages } from 'react-chat-widget';
import 'react-chat-widget/lib/styles.css';
import './OttoChatWidget.css';
import useSupabase from '../hooks/useSupabase';

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

interface ChatPayload {
  query: string;
  conversation_history: { query: string; response: string }[];
  user_id: string;
  message_id: string;
}

const OttoChatWidget = () => {
  const [chatWindowOpen, setChatWindowOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const welcomeMessageSent = useRef(false);
  const processingMessageId = useRef<number | null>(null);
  const { user } = useSupabase();

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
      const payload: ChatPayload = {
        query: newMessage,
        conversation_history: messages.map(msg => ({ 
          query: msg.query, 
          response: msg.response 
        })),
        user_id: user?.id || 'anonymous',
        message_id: messageId
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
      )}
    />
  );
};

export default OttoChatWidget;