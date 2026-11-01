import { useControlPanelStore } from "@/app/(platform)/build/stores/controlPanelStore";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/molecules/Popover/Popover";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import { ControlPanelButton } from "../../ControlPanelButton";
import { BlockMenuContent } from "../BlockMenuContent/BlockMenuContent";
import { ToyBrickIcon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

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
              <Icon icon={ToyBrickIcon} className="size-5" />
            </ControlPanelButton>
          </PopoverTrigger>
        </TooltipTrigger>
        <TooltipContent side="right">Blocks</TooltipContent>
      </Tooltip>

      {/* rounded-medium is the 12px corner this menu renders today, the old
          popover's own corner. */}
      <PopoverContent
        side="right"
        align="start"
        sideOffset={16}
        className="absolute h-[80vh] w-[46.625rem] overflow-hidden rounded-medium border-none border-border-base p-0 text-foreground-base shadow-sm"
        data-id="blocks-control-popover-content"
      >
        <BlockMenuContent />
      </PopoverContent>
    </Popover>
  );
};
