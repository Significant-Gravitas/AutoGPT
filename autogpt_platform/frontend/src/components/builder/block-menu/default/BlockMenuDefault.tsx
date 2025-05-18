import React, { useState } from "react";
import BlockMenuSidebar from "./BlockMenuSidebar";
import { Separator } from "@/components/ui/separator";
import BlockMenuDefaultContent from "./BlockMenuDefaultContent";

export type DefaultStateType =
  | "suggestion"
  | "all_blocks"
  | "input_blocks"
  | "action_blocks"
  | "output_blocks"
  | "integrations"
  | "marketplace_agents"
  | "my_agents";

const BlockMenuDefault: React.FC = () => {
  const [defaultState, setDefaultState] =
    useState<DefaultStateType>("suggestion");
  const [integration, setIntegration] = useState("");

  return (
    <div className="flex flex-1 overflow-y-auto">
      {/* Left sidebar */}
      <BlockMenuSidebar
        defaultState={defaultState}
        setDefaultState={setDefaultState}
        setIntegration={setIntegration}
      />

      <Separator className="h-full w-[1px] text-zinc-300" />

      <BlockMenuDefaultContent
        defaultState={defaultState}
        setDefaultState={setDefaultState}
        setIntegration={setIntegration}
        integration={integration}
      />
    </div>
  );
};

export default BlockMenuDefault;
