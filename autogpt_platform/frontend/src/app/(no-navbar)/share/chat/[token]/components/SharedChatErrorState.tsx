import { InfoIcon } from "@phosphor-icons/react";

type Props = {
  reason?: string;
  onRetry: () => void;
};

export function SharedChatErrorState({ onRetry }: Props) {
  return (
    <div className="flex h-full w-full flex-1 items-center justify-center">
      <div className="mx-auto w-full max-w-md p-6">
        <div className="space-y-4 rounded-lg border border-dashed border-zinc-300 p-6 text-center">
          <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-full bg-zinc-100">
            <InfoIcon size={24} className="text-zinc-500" />
          </div>
          <div className="space-y-2">
            <h3 className="text-lg font-semibold">Share link not found</h3>
            <p className="text-sm text-zinc-500">
              This link is invalid or has been disabled by the owner. Ask the
              person who shared it for an updated link.
            </p>
          </div>
          <button
            onClick={onRetry}
            className="text-sm text-zinc-700 underline hover:text-zinc-900"
          >
            Try again
          </button>
        </div>
        <p className="mt-8 text-center text-xs text-zinc-400">
          Powered by AutoGPT Platform
        </p>
      </div>
    </div>
  );
}
