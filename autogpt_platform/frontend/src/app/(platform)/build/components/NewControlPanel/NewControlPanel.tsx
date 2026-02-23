import { cn } from "@/lib/utils";
import React, { memo } from "react";
import { BlockMenu } from "./NewBlockMenu/BlockMenu/BlockMenu";
import { useNewControlPanel } from "./useNewControlPanel";
import { Separator } from "@/components/__legacy__/ui/separator";
import { NewSaveControl } from "./NewSaveControl/NewSaveControl";
import { UndoRedoButtons } from "./UndoRedoButtons";

export const NewControlPanel = memo(() => {
  useNewControlPanel({});

  return (
    <section
      className={cn(
        "absolute left-4 top-10 z-10 overflow-hidden rounded-[1rem] border-none bg-white p-0 shadow-[0_1px_5px_0_rgba(0,0,0,0.1)]",
      )}
    >
      <div className="flex flex-col items-center justify-center rounded-[1rem] p-0">
        <BlockMenu />
        <Separator className="text-[#E1E1E1]" />
        <NewSaveControl />
        <Separator className="text-[#E1E1E1]" />
        <UndoRedoButtons />
      </div>
    </section>
  );
});

export default NewControlPanel;

NewControlPanel.displayName = "NewControlPanel";
