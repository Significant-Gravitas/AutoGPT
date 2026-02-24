import { useControlPanelStore } from "@/app/(platform)/build/stores/controlPanelStore";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/__legacy__/ui/popover";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import { LegoIcon } from "@phosphor-icons/react";
import { ControlPanelButton } from "../../ControlPanelButton";
import { BlockMenuContent } from "../BlockMenuContent/BlockMenuContent";

export const BlockMenu = () => {
  const { blockMenuOpen, setBlockMenuOpen, forceOpenBlockMenu } =
    useControlPanelStore();
  return (
    <Popover
      onOpenChange={(open) => {
        if (!forceOpenBlockMenu || open) {
          setBlockMenuOpen(open);
        }
      }}
      open={forceOpenBlockMenu ? true : blockMenuOpen}
    >
      <Tooltip delayDuration={100}>
        <TooltipTrigger asChild>
          <PopoverTrigger className="hover:cursor-pointer">
            <ControlPanelButton
              data-id="blocks-control-popover-trigger"
              data-testid="blocks-control-blocks-button"
              selected={blockMenuOpen}
              className="rounded-none"
            >
              <LegoIcon className="size-5" />
            </ControlPanelButton>
          </PopoverTrigger>
        </TooltipTrigger>
        <TooltipContent side="right">Blocks</TooltipContent>
      </Tooltip>

      <PopoverContent
        side="right"
        align="start"
        sideOffset={16}
        className="absolute h-[80vh] w-[46.625rem] overflow-hidden rounded-[1rem] border-none p-0 shadow-[0_2px_6px_0_rgba(0,0,0,0.05)]"
        data-id="blocks-control-popover-content"
      >
        <BlockMenuContent />
      </PopoverContent>
    </Popover>
  );
};
