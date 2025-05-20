import React from "react";
import MenuItem from "../MenuItem";
import { DefaultStateType, useBlockMenuContext } from "../block-menu-provider";

const BlockMenuSidebar: React.FC = ({}) => {
  const { defaultState, setDefaultState, setIntegration } =
    useBlockMenuContext();

  // TEMPORARY FETCHING
  const topLevelMenuItems = [
    {
      name: "Suggestion",
      type: "suggestion",
    },
    {
      name: "All blocks",
      type: "all_blocks",
      number: 103,
    },
  ];

  const subMenuItems = [
    {
      name: "Input blocks",
      type: "input_blocks",
      number: 12,
    },
    {
      name: "Action blocks",
      type: "action_blocks",
      number: 40,
    },
    {
      name: "Output blocks",
      type: "output_blocks",
      number: 6,
    },
  ];

  const bottomMenuItems = [
    {
      name: "Integrations",
      type: "integrations",
      number: 24,
      onClick: () => {
        setIntegration("");
        setDefaultState("integrations");
      },
    },
    {
      name: "Marketplace Agents",
      type: "marketplace_agents",
      number: 103,
    },
    {
      name: "My Agents",
      type: "my_agents",
      number: 6,
    },
  ];

  return (
    <div className="space-y-2 px-4 pt-4">
      {topLevelMenuItems.map((item) => (
        <MenuItem
          key={item.type}
          name={item.name}
          number={item.number}
          selected={defaultState === item.type}
          onClick={() => setDefaultState(item.type as DefaultStateType)}
        />
      ))}
      <div className="ml-[0.5365rem] border-l border-black/10 pl-[0.75rem]">
        {subMenuItems.map((item) => (
          <MenuItem
            key={item.type}
            name={item.name}
            number={item.number}
            selected={defaultState === item.type}
            onClick={() => setDefaultState(item.type as DefaultStateType)}
          />
        ))}
      </div>
      {bottomMenuItems.map((item) => (
        <MenuItem
          key={item.type}
          name={item.name}
          number={item.number}
          selected={defaultState === item.type}
          onClick={
            item.onClick ||
            (() => setDefaultState(item.type as DefaultStateType))
          }
        />
      ))}
    </div>
  );
};

export default BlockMenuSidebar;
