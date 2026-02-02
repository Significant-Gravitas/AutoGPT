import { ChatSidebar } from "../ChatSidebar/ChatSidebar";

interface EmptySessionProps {
  isCreating: boolean;
  handleSubmit: (e: React.FormEvent) => void;
  input: string;
  setInput: (input: string) => void;
}

export const EmptySession = ({ isCreating, handleSubmit, input, setInput }: EmptySessionProps) => {
  return (
    <div className="flex h-full flex-1 flex-col items-center justify-center bg-zinc-100 p-4">
      <h2 className="mb-4 text-xl font-semibold text-zinc-700">
        Start a new conversation
      </h2>
      <form onSubmit={handleSubmit} className="w-full max-w-md">
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
          {isCreating ? "Starting..." : "Start Chat"}
        </button>
      </form>
    </div>
  );
};