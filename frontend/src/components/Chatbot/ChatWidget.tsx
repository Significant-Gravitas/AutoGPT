import { useState, useEffect, useRef } from 'react';
import { MessageCircle, X, Loader2 } from 'lucide-react';
import { v4 as uuidv4 } from 'uuid';
import ChatMessage from './ChatMessage';
import ChatInput from './ChatInput';
import { sendChatMessage } from '../../services/api';

interface Message {
  id: string;
  message: string;
  role: 'user' | 'assistant';
}

const STORAGE_KEY = 'neus_chat_session_id';

export default function ChatWidget() {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState<string>('');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Initialize session ID
  useEffect(() => {
    let storedSessionId = localStorage.getItem(STORAGE_KEY);
    if (!storedSessionId) {
      storedSessionId = uuidv4();
      localStorage.setItem(STORAGE_KEY, storedSessionId);
    }
    setSessionId(storedSessionId);
  }, []);

  // Add welcome message when chat opens for the first time
  useEffect(() => {
    if (isOpen && messages.length === 0) {
      setMessages([{
        id: uuidv4(),
        message: '¡Hola! Soy el asistente virtual de NEUS. ¿En qué puedo ayudarte hoy? Puedo responder preguntas sobre nuestros servicios, sectores especializados, o ayudarte a agendar un diagnóstico gratuito.',
        role: 'assistant'
      }]);
    }
  }, [isOpen, messages.length]);

  // Scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSendMessage = async (message: string) => {
    // Add user message
    const userMessage: Message = {
      id: uuidv4(),
      message,
      role: 'user'
    };
    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);

    try {
      const response = await sendChatMessage(message, sessionId);

      // Add assistant response
      const assistantMessage: Message = {
        id: uuidv4(),
        message: response.response,
        role: 'assistant'
      };
      setMessages(prev => [...prev, assistantMessage]);

      // Update session ID if it changed
      if (response.session_id !== sessionId) {
        setSessionId(response.session_id);
        localStorage.setItem(STORAGE_KEY, response.session_id);
      }
    } catch (error) {
      // Add error message
      const errorMessage: Message = {
        id: uuidv4(),
        message: 'Lo siento, ha ocurrido un error. Por favor, intenta nuevamente o contáctanos directamente.',
        role: 'assistant'
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <>
      {/* Chat Window */}
      {isOpen && (
        <div className="fixed bottom-24 right-4 sm:right-6 w-[90vw] sm:w-96 h-[600px] max-h-[80vh] bg-white rounded-2xl shadow-2xl flex flex-col z-50 border-2 border-gray-200">
          {/* Header */}
          <div className="bg-gradient-to-r from-primary-600 to-secondary-600 p-4 rounded-t-2xl flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-white rounded-full flex items-center justify-center">
                <MessageCircle className="h-6 w-6 text-primary-600" />
              </div>
              <div>
                <h3 className="font-bold text-white">Chat con NEUS</h3>
                <p className="text-xs text-white/80">Siempre aquí para ayudarte</p>
              </div>
            </div>
            <button
              onClick={() => setIsOpen(false)}
              className="p-2 hover:bg-white/20 rounded-lg transition-colors"
            >
              <X className="h-5 w-5 text-white" />
            </button>
          </div>

          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-gray-50">
            {messages.map((msg) => (
              <ChatMessage key={msg.id} message={msg.message} role={msg.role} />
            ))}

            {/* Loading indicator */}
            {isLoading && (
              <div className="flex items-center space-x-2 text-gray-500">
                <Loader2 className="h-4 w-4 animate-spin" />
                <span className="text-sm">NEUS está escribiendo...</span>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Input */}
          <ChatInput onSendMessage={handleSendMessage} disabled={isLoading} />
        </div>
      )}

      {/* Floating Button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="fixed bottom-6 right-4 sm:right-6 w-14 h-14 bg-gradient-to-r from-primary-600 to-secondary-600 text-white rounded-full shadow-lg hover:shadow-xl hover:scale-110 transition-all duration-200 flex items-center justify-center z-50"
        aria-label="Abrir chat"
      >
        {isOpen ? (
          <X className="h-6 w-6" />
        ) : (
          <MessageCircle className="h-6 w-6" />
        )}
      </button>
    </>
  );
}
