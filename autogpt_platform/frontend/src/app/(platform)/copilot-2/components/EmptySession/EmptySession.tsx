interface Props {
  isCreating: boolean;
  onCreateSession: (e: React.FormEvent) => void;
}

export function EmptySession({ isCreating, onCreateSession }: Props) {
  return (
    <div className="flex h-full flex-1 flex-col items-center justify-center bg-zinc-100 p-4">
      <h2 className="mb-4 text-xl font-semibold text-zinc-700">
        Start a new conversation
      </h2>
      <form onSubmit={onCreateSession} className="w-full max-w-md">
        <button
          type="submit"
          disabled={isCreating}
          className="w-full rounded-md bg-blue-600 px-4 py-2 text-white transition-colors hover:bg-blue-700 disabled:opacity-50"
        >
          {isCreating ? "Creating..." : "Start New Chat"}
        </button>
      </form>
    </div>
  );
}