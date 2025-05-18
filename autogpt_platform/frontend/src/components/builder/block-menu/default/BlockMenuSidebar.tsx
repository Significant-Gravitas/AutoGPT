import React from "react";
import MenuItem from "../MenuItem";
import { DefaultStateType } from "./BlockMenuDefault";

interface BlockMenuSidebarProps {
  defaultState: DefaultStateType;
  setDefaultState: React.Dispatch<React.SetStateAction<DefaultStateType>>;
  setIntegration: React.Dispatch<React.SetStateAction<string>>;
}

const BlockMenuSidebar: React.FC<BlockMenuSidebarProps> = ({
  defaultState,
  setDefaultState,
  setIntegration,
}) => {
  // Update Block Menu fetching
  return (
    <div className="space-y-2 p-4">
      <MenuItem
        name={"Suggestion"}
        selected={defaultState == "suggestion"}
        onClick={() => setDefaultState("suggestion")}
      />
      <MenuItem
        name={"All blocks"}
        number={103}
        selected={defaultState == "all_blocks"}
        onClick={() => setDefaultState("all_blocks")}
      />
      <div className="ml-[0.5365rem] border-l border-black/10 pl-[0.75rem]">
        <MenuItem
          name={"Input blocks"}
          number={12}
          selected={defaultState == "input_blocks"}
          onClick={() => setDefaultState("input_blocks")}
        />
        <MenuItem
          name={"Action blocks"}
          number={40}
          selected={defaultState == "action_blocks"}
          onClick={() => setDefaultState("action_blocks")}
        />
        <MenuItem
          name={"Output blocks"}
          number={6}
          selected={defaultState == "output_blocks"}
          onClick={() => setDefaultState("output_blocks")}
        />
      </div>
      <MenuItem
        name={"Integrations"}
        number={24}
        selected={defaultState == "integrations"}
        onClick={() => {
          setIntegration("");
          setDefaultState("integrations");
        }}
      />
      <MenuItem
        name={"Marketplace Agents"}
        number={103}
        selected={defaultState == "marketplace_agents"}
        onClick={() => setDefaultState("marketplace_agents")}
      />
      <MenuItem
        name={"My Agents"}
        number={6}
        selected={defaultState == "my_agents"}
        onClick={() => setDefaultState("my_agents")}
      />
    </div>
  );
};

export default BlockMenuSidebar;
