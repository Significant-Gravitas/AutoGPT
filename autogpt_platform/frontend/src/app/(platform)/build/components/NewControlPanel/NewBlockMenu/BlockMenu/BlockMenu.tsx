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
          popover's own corner.
          TODO(tokens): today's popover look. No exact role: border-neutral-200
          #E5E5E5; text-neutral-950 #0A0A0A is the `focus` role, a ring
          colour, not a text colour (SECRT-3154). */}
      <PopoverContent
        side="right"
        align="start"
        sideOffset={16}
        // TODO(tokens): no exact role: shadow-[…] #000000
        className="absolute h-[80vh] w-[46.625rem] overflow-hidden rounded-medium border-none border-neutral-200 p-0 text-neutral-950 shadow-[0_2px_6px_0_rgba(0,0,0,0.05)] dark:border-neutral-800 dark:bg-neutral-950 dark:text-neutral-50"
        data-id="blocks-control-popover-content"
      >
        <BlockMenuContent />
      </PopoverContent>
    </Popover>
  );
};
