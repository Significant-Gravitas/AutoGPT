import { Card, CardContent } from "@/components/ui/card";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { cn } from "@/lib/utils";
import React from "react";

/**
 * Represents a control element for the ControlPanel Component.
 * @type {Object} Control
 * @property {React.ReactNode} icon - The icon of the control from lucide-react https://lucide.dev/icons/
 * @property {string} label - The label of the control, to be leveraged by ToolTip.
 * @property {onclick} onClick - The function to be executed when the control is clicked.
 */
export type Control = {
  icon: React.ReactNode;
  label: string;
  disabled?: boolean;
  onClick: () => void;
};

interface ControlPanelProps {
  controls: Control[];
  topChildren?: React.ReactNode;
  botChildren?: React.ReactNode;
  className?: string;
}

/**
 * ControlPanel component displays a panel with controls as icons.tsx with the ability to take in children.
 * @param {Object} ControlPanelProps - The properties of the control panel component.
 * @param {Array} ControlPanelProps.controls - An array of control objects representing actions to be preformed.
 * @param {Array} ControlPanelProps.children - The child components of the control panel.
 * @param {string} ControlPanelProps.className - Additional CSS class names for the control panel.
 * @returns The rendered control panel component.
 */
export const ControlPanel = ({
  controls,
  topChildren,
  botChildren,
  className,
}: ControlPanelProps) => {
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
};
export default ControlPanel;
