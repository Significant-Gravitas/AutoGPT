import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import React from "react";

export type Control = {
  icon: React.ReactNode;
  label: string;
  disabled?: boolean;
  onClick: () => void;
};

interface Props {
  controls: Control[];
  topChildren?: React.ReactNode;
  botChildren?: React.ReactNode;
  className?: string;
}

export function ControlPanel({
  controls,
  topChildren,
  botChildren,
  className,
}: Props) {
  return (
    <Card className={cn("m-4 mt-24 w-14 dark:bg-slate-900", className)}>
      <CardContent className="p-0">
        <div className="flex flex-col items-center gap-3 rounded-xl py-3">
          {topChildren}
          <Separator className="dark:bg-slate-700" />
          {controls.map((control, index) => (
            <Tooltip key={index} delayDuration={500}>
              <TooltipTrigger asChild>
                <div>
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={() => control.onClick()}
                    data-id={`control-button-${index}`}
                    data-testid={`blocks-control-${control.label.toLowerCase()}-button`}
                    disabled={control.disabled || false}
                    className="dark:bg-slate-900 dark:text-slate-100 dark:hover:bg-slate-800"
                  >
                    {control.icon}
                    <span className="sr-only">{control.label}</span>
                  </Button>
                </div>
              </TooltipTrigger>
              <TooltipContent
                side="right"
                className="dark:bg-slate-800 dark:text-slate-100"
              >
                {control.label}
              </TooltipContent>
            </Tooltip>
          ))}
          <Separator className="dark:bg-slate-700" />
          {botChildren}
        </div>
      </CardContent>
    </Card>
  );
}
export default ControlPanel;
