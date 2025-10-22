import { Button } from "@/components/atoms/Button/Button";
import { PlayIcon } from "lucide-react";
import { useRunGraph } from "./useRunGraph";
import { useGraphStore } from "@/app/(platform)/build/stores/graphStore";
import { useShallow } from "zustand/react/shallow";

export const RunGraph = () => {
  const { runGraph, isSaving } = useRunGraph();
  const isGraphRunning = useGraphStore(
    useShallow((state) => state.isGraphRunning),
  );

  return (
    <Button
      variant="primary"
      size="large"
      className="relative min-w-44 bg-purple-500 text-lg"
      onClick={() => runGraph()}
      disabled={isGraphRunning || isSaving}
      loading={isGraphRunning || isSaving}
    >
      {!isGraphRunning && !isSaving && <PlayIcon className="mr-1 size-5" />}
      {isGraphRunning || isSaving ? "Running..." : "Run Graph"}
    </Button>
  );
};
