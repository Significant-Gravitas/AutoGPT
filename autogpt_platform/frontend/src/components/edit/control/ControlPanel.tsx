import { Separator } from "@/components/ui/separator";
import { cn } from "@/lib/utils";
import React from "react";
import ControlPanelButton from "@/components/builder/block-menu/ControlPanelButton";

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
    <section
      className={cn(
        "absolute left-4 top-24 z-10 w-[4.25rem] overflow-hidden rounded-[1rem] border-none bg-white p-0 shadow-[0_1px_5px_0_rgba(0,0,0,0.1)]",
        className,
      )}
    >
      <div className="flex flex-col items-center justify-center rounded-[1rem] p-0">
        {topChildren}
        <Separator className="text-[#E1E1E1]" />
        {controls.map((control, index) => (
          <ControlPanelButton
            key={index}
            onClick={() => control.onClick()}
            data-id={`control-button-${index}`}
            data-testid={`blocks-control-${control.label.toLowerCase()}-button`}
            disabled={control.disabled || false}
            className="rounded-none"
          >
            {control.icon}
          </ControlPanelButton>
        ))}
        <Separator className="text-[#E1E1E1]" />
        {botChildren}
      </div>
    </section>
  );
};
export default ControlPanel;
