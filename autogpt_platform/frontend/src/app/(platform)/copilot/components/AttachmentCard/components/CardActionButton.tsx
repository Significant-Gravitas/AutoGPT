import { Button } from "@/components/atoms/Button/Button";
import {
  Tooltip,
  TooltipContent,
  TooltipPortal,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import type { IconSvgElement } from "@hugeicons/react";
import type { ReactNode } from "react";

interface Props {
  label: string;
  onClick: () => void;
  size: "xs" | "icon-xs";
  leadingIcon?: IconSvgElement;
  className?: string;
  children: ReactNode;
}

// The card sits inside the composer's <form> and its scrolling, clipped
// attachment list: the button must not submit, and its tooltip is portalled.
export function CardActionButton({
  label,
  onClick,
  size,
  leadingIcon,
  className,
  children,
}: Props) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Button
          type="button"
          variant="ghost"
          size={size}
          leadingIcon={leadingIcon}
          aria-label={label}
          withTooltip={false}
          onClick={onClick}
          className={className}
        >
          {children}
        </Button>
      </TooltipTrigger>
      <TooltipPortal>
        <TooltipContent collisionPadding={8} className="break-words">
          {label}
        </TooltipContent>
      </TooltipPortal>
    </Tooltip>
  );
}
