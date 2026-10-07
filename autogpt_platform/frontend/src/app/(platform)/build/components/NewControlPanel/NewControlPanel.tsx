import { cn } from "@/lib/utils";
import { memo } from "react";
import { BlockMenu } from "./NewBlockMenu/BlockMenu/BlockMenu";
import { Separator } from "@/components/ui/separator";
import { NewSaveControl } from "./NewSaveControl/NewSaveControl";
import { GraphSearchMenu } from "./NewSearchGraph/GraphMenu/GraphMenu";
import { UndoRedoButtons } from "./UndoRedoButtons";
import { useGraphSearchShortcut } from "./useGraphSearchShortcut";

interface Props {
  isReadOnly?: boolean;
}

export const NewControlPanel = memo(function NewControlPanel({
  isReadOnly = false,
}: Props) {
  useGraphSearchShortcut();

  return (
    <section
      className={cn(
        "absolute left-4 top-10 z-10 overflow-hidden rounded-2xl border-none bg-white p-0 shadow-[0_1px_5px_0_rgba(0,0,0,0.1)]",
      )}
    >
      <div className="flex flex-col items-center justify-center rounded-2xl p-0">
        {isReadOnly ? (
          <GraphSearchMenu />
        ) : (
          <>
            <BlockMenu />
            <Separator className="text-zinc-200" />
            <GraphSearchMenu />
            <Separator className="text-zinc-200" />
            <NewSaveControl />
            <Separator className="text-zinc-200" />
            <UndoRedoButtons />
          </>
        )}
      </div>
    </section>
  );
});

export default NewControlPanel;
