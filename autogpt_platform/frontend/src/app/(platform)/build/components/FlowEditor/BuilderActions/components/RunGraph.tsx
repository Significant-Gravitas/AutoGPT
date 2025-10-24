import { Button } from "@/components/atoms/Button/Button";
import { PlayIcon } from "lucide-react";
import { useRunGraph } from "./useRunGraph";
import { useGraphStore } from "@/app/(platform)/build/stores/graphStore";
import { useShallow } from "zustand/react/shallow";
import { StopIcon } from "@phosphor-icons/react";
import { cn } from "@/lib/utils";

export const RunGraph = () => {
  const { runGraph, isSaving } = useRunGraph();
  const isGraphRunning = useGraphStore(
    useShallow((state) => state.isGraphRunning),
  );

  return (
    <Button
      variant="primary"
      size="large"
      className={cn(
        "relative min-w-44 border-none bg-gradient-to-r from-purple-500 to-pink-500 text-lg",
      )}
      onClick={() => runGraph()}
    >
      {!isGraphRunning && !isSaving ? (
        <PlayIcon className="mr-1 size-5" />
      ) : (
        <StopIcon className="mr-1 size-5" />
      )}
      {isGraphRunning || isSaving ? "Stop Agent" : "Run Agent"}
    </Button>
  );
};
