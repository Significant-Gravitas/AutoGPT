import { Card, CardContent } from "@/components/ui/card";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
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
  onClick: () => void;
};

interface ControlPanelProps {
  controls: Control[];
  children?: React.ReactNode;
}

/**
 * ControlPanel component displays a panel with controls as icons with the ability to take in children.
 * @param {Object} ControlPanelProps - The properties of the control panel component.
 * @param {Array} ControlPanelProps.controls - An array of control objects representing actions to be preformed.
 * @param {Array} ControlPanelProps.children - The child components of the control panel.
 * @returns The rendered control panel component.
 */
export const ControlPanel = ({ controls, children }: ControlPanelProps) => {
  return (
    <aside className="hidden w-14 flex-col sm:flex">
      <Card>
        <CardContent className="p-0">
          <div className="flex flex-col items-center gap-4 px-2 sm:py-5 rounded-radius">
            {controls.map((control, index) => (
              <Tooltip key={index} delayDuration={500}>
                <TooltipTrigger asChild>
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={() => control.onClick()}
                  >
                    {control.icon}
                    <span className="sr-only">{control.label}</span>
                  </Button>
                </TooltipTrigger>
                <TooltipContent side="right">{control.label}</TooltipContent>
              </Tooltip>
            ))}
            <Separator />
            {children}
          </div>
        </CardContent>
      </Card>
    </aside>
  );
};
export default ControlPanel;
