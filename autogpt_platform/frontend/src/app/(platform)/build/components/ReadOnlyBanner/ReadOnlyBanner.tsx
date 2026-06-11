import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { CopyIcon, XIcon } from "@phosphor-icons/react";
import { useState } from "react";
import { useDuplicateGraph } from "../../hooks/useDuplicateGraph";

export function ReadOnlyBanner() {
  const { duplicate, isDuplicating, canDuplicate, isCheckingLibrary } =
    useDuplicateGraph();
  const [isDismissed, setIsDismissed] = useState(false);

  if (isDismissed) return null;

  // Show the Duplicate CTA immediately while we look up the library agent, only
  // falling back to the "add to library" hint once we know it can't be forked.
  const showDuplicate = canDuplicate || isCheckingLibrary;

  return (
    <div
      data-id="read-only-banner"
      role="status"
      aria-live="polite"
      className="absolute left-1/2 top-4 z-20 flex -translate-x-1/2 select-none items-center gap-3 rounded-full bg-white px-4 py-2 shadow-lg"
    >
      <Text variant="body" className="px-2 text-zinc-700">
        You&apos;re viewing a read-only copy of this agent.
        <br />
        {showDuplicate
          ? "Duplicate it to make changes."
          : "Add it to your library to enable duplication."}
      </Text>
      {showDuplicate && (
        <Button
          variant="primary"
          size="small"
          onClick={duplicate}
          loading={isDuplicating}
          disabled={!canDuplicate}
          leftIcon={<CopyIcon className="size-4" />}
        >
          Duplicate
        </Button>
      )}
      <Button
        variant="ghost"
        size="icon"
        onClick={() => setIsDismissed(true)}
        aria-label="Dismiss"
        title="Dismiss"
      >
        <XIcon className="h-4 w-4" />
      </Button>
    </div>
  );
}
