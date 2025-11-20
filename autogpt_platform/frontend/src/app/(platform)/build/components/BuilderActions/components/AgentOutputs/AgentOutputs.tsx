import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/__legacy__/ui/sheet";
import { BuilderActionButton } from "../BuilderActionButton";
import { BookOpenIcon } from "@phosphor-icons/react";

export const AgentOutputs = ({ flowID }: { flowID: string | null }) => {
  return (
    <>
      <Sheet>
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <SheetTrigger asChild>
                <BuilderActionButton disabled={!flowID}>
                  <BookOpenIcon className="size-6" />
                </BuilderActionButton>
              </SheetTrigger>
            </TooltipTrigger>
            <TooltipContent>
              <p>Agent Outputs</p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
        <SheetContent>
          <SheetHeader>
            <SheetTitle>Agent Outputs</SheetTitle>
            <SheetDescription>
              View and manage your agent outputs here.
            </SheetDescription>
          </SheetHeader>
        </SheetContent>
      </Sheet>
    </>
  );
};
