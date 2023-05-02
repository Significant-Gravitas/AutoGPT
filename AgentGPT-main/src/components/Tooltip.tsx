import React from "react";
import * as TooltipPrimitive from "@radix-ui/react-tooltip";
import type { toolTipProperties } from "./types";

interface TooltipProps {
  child: React.ReactNode;
  toolTipProperties?: toolTipProperties;
  style?: { [key: string]: string };
  sideOffset: number;
}

const Tooltip = ({
  child,
  toolTipProperties = { message: "", disabled: true },
  style = { container: "" },
  sideOffset,
}: TooltipProps) => {
  const { message, disabled } = toolTipProperties;
  return (
    <div className={style.container}>
      <TooltipPrimitive.Provider>
        <TooltipPrimitive.Root delayDuration={0}>
          <TooltipPrimitive.Trigger asChild>{child}</TooltipPrimitive.Trigger>
          {disabled ? null : (
            <TooltipPrimitive.Portal>
              <TooltipPrimitive.Content
                className="will-change animation-transform user-select-none z-40 w-3/5 rounded-sm bg-black px-3.5 py-2.5 text-white shadow-lg "
                sideOffset={sideOffset}
              >
                {message}
                <TooltipPrimitive.Arrow className="fill-black" />
              </TooltipPrimitive.Content>
            </TooltipPrimitive.Portal>
          )}
        </TooltipPrimitive.Root>
      </TooltipPrimitive.Provider>
    </div>
  );
};

export default Tooltip;
