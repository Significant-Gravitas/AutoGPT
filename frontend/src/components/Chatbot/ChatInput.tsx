import { useState, FormEvent, KeyboardEvent } from 'react';
import { Send } from 'lucide-react';

interface ChatInputProps {
  onSendMessage: (message: string) => void;
  disabled?: boolean;
}

export default function ChatInput({ onSendMessage, disabled = false }: ChatInputProps) {
  const [message, setMessage] = useState('');

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (message.trim() && !disabled) {
      onSendMessage(message.trim());
      setMessage('');
    }
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="p-4 border-t border-gray-200 bg-white">
      <div className="flex items-end space-x-2">
        <textarea
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Escribe tu mensaje..."
          disabled={disabled}
          rows={1}
          className="flex-1 px-4 py-2 border-2 border-gray-300 rounded-lg focus:border-primary-600 focus:outline-none resize-none disabled:opacity-50 disabled:cursor-not-allowed"
          style={{ maxHeight: '100px' }}
        />
        <button
          type="submit"
          disabled={!message.trim() || disabled}
          className="p-3 bg-gradient-to-r from-primary-600 to-secondary-600 text-white rounded-lg hover:shadow-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed flex-shrink-0"
        >
          <Send className="h-5 w-5" />
        </button>
      </div>
      <p className="text-xs text-gray-500 mt-2">
        Presiona Enter para enviar, Shift+Enter para nueva línea
      </p>
    </form>
  );
}
