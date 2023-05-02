import React from "react";
import Tooltip from "./Tooltip";
import type { toolTipProperties } from "./types";

interface LabelProps {
  left?: React.ReactNode;
  type?: string;
  toolTipProperties?: toolTipProperties;
}

const Label = ({ type, left, toolTipProperties }: LabelProps) => {
  const isTypeTextArea = () => {
    return type === "textarea";
  };

  return (
    <Tooltip
      child={
        <div
          className={`center flex items-center rounded-xl rounded-r-none ${
            type !== "range" ? "border-r-0 border-white/10 md:border-[2px]" : ""
          }  py-2 text-sm font-semibold tracking-wider transition-all sm:py-3 md:pl-3 md:text-lg
          ${isTypeTextArea() ? "md:h-20" : ""}`}
        >
          {left}
        </div>
      }
      style={{
        container: `md:w-1/4`,
      }}
      sideOffset={0}
      toolTipProperties={toolTipProperties}
    ></Tooltip>
  );
};

export default Label;
