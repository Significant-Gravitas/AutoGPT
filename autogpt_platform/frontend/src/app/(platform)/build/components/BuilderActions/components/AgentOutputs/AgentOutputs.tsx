import { Button } from "@/components/atoms/Button/Button";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import { LogOutIcon } from "lucide-react";

export const AgentOutputs = () => {
  return (
    <>
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger asChild>
            {/* Todo: Implement Agent Outputs */}
            <Button
              variant="primary"
              size="large"
              className={"relative min-w-0 border-none text-lg"}
            >
              <LogOutIcon className="size-6" />
            </Button>
          </TooltipTrigger>
          <TooltipContent>
            <p>Agent Outputs</p>
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>
    </>
  );
};
