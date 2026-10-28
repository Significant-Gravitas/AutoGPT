import { cn } from "@/lib/utils";
import { memo } from "react";
import { BlockMenu } from "./NewBlockMenu/BlockMenu/BlockMenu";
import { Separator } from "@/components/atoms/Separator/Separator";
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
        // TODO(tokens): no exact role: shadow-[…] #000000
        "absolute left-4 top-10 z-10 overflow-hidden rounded-[1rem] border-none bg-surface p-0 shadow-[0_1px_5px_0_rgba(0,0,0,0.1)]",
      )}
    >
      <div className="flex flex-col items-center justify-center rounded-[1rem] p-0">
        {isReadOnly ? (
          <GraphSearchMenu />
        ) : (
          <>
            <BlockMenu />
            <Separator
              // TODO(tokens): no exact role: text-[#E1E1E1] #E1E1E1
              className="text-[#E1E1E1]"
            />
            <GraphSearchMenu />
            <Separator
              // TODO(tokens): no exact role: text-[#E1E1E1] #E1E1E1
              className="text-[#E1E1E1]"
            />
            <NewSaveControl />
            <Separator
              // TODO(tokens): no exact role: text-[#E1E1E1] #E1E1E1
              className="text-[#E1E1E1]"
            />
            <UndoRedoButtons />
          </>
        )}
      </div>
    </section>
  );
});

export default NewControlPanel;
