import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import { BuilderActionButton } from "../BuilderActionButton";
import { BookOpenIcon } from "@phosphor-icons/react";

export const AgentOutputs = ({ flowID }: { flowID: string | null }) => {
  return (
    <>
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger asChild>
            {/* Todo: Implement Agent Outputs */}
            <BuilderActionButton disabled={!flowID}>
              <BookOpenIcon className="size-6" />
            </BuilderActionButton>
          </TooltipTrigger>
          <TooltipContent>
            <p>Agent Outputs</p>
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>
    </>
  );
};
