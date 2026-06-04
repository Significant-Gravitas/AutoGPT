import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { CopyIcon } from "@phosphor-icons/react";
import { useDuplicateGraph } from "../../hooks/useDuplicateGraph";

export function ReadOnlyBanner() {
  const { duplicate, isDuplicating, canDuplicate } = useDuplicateGraph();

  return (
    <div
      data-id="read-only-banner"
      role="status"
      aria-live="polite"
      className="absolute left-1/2 top-4 z-20 flex -translate-x-1/2 select-none items-center gap-3 rounded-full bg-white px-4 py-2 shadow-lg"
    >
      <Text variant="body" className="text-zinc-700">
        {canDuplicate
          ? "You're viewing a read-only copy of this agent. Duplicate it to make changes."
          : "You're viewing a read-only copy of this agent. Add it to your library to enable duplication."}
      </Text>
      {canDuplicate && (
        <Button
          variant="primary"
          size="small"
          onClick={duplicate}
          loading={isDuplicating}
          leftIcon={<CopyIcon className="size-4" />}
        >
          Duplicate
        </Button>
      )}
    </div>
  );
}
