import React from "react";
import { MenuItem } from "../MenuItem";
import { DefaultStateType } from "../block-menu-provider";
import { useBlockMenuSidebar } from "./useBlockMenuSidebar";
import { Skeleton } from "@/components/ui/skeleton";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";

export const BlockMenuSidebar = () => {
  const { blockCounts, setDefaultState, defaultState, isLoading, isError, error } = useBlockMenuSidebar();

  if (isLoading) {
    return (
      <div className="w-fit space-y-2 px-4 pt-4">
        <Skeleton className="h-12 w-[12.875rem]" />
        <Skeleton className="h-12 w-[12.875rem]" />
        <Skeleton className="h-12 w-[12.875rem]" />
        <Skeleton className="h-12 w-[12.875rem]" />
        <Skeleton className="h-12 w-[12.875rem]" />
        <Skeleton className="h-12 w-[12.875rem]" />
      </div>
    );
  }
  if (isError) {
    return <div className="w-fit space-y-2 px-4 pt-4">
      <ErrorCard className="w-[12.875rem]" httpError={{status: 500, statusText: "Internal Server Error", message: error?.detail || 'An error occurred'}} />
      </div>
  }

  const topLevelMenuItems = [
    {
      name: "Suggestion",
      type: "suggestion",
    },
    {
      name: "All blocks",
      type: "all_blocks",
      number: blockCounts?.all_blocks,
    },
  ];

  const subMenuItems = [
    {
      name: "Input blocks",
      type: "input_blocks",
      number: blockCounts?.input_blocks,
    },
    {
      name: "Action blocks",
      type: "action_blocks",
      number: blockCounts?.action_blocks,
    },
    {
      name: "Output blocks",
      type: "output_blocks",
      number: blockCounts?.output_blocks,
    },
  ];

  const bottomMenuItems = [
    {
      name: "Integrations",
      type: "integrations",
      number: blockCounts?.integrations,
      onClick: () => {
        setDefaultState("integrations");
      },
    },
    {
      name: "Marketplace Agents",
      type: "marketplace_agents",
      number: blockCounts?.marketplace_agents,
    },
    {
      name: "My Agents",
      type: "my_agents",
      number: blockCounts?.my_agents,
    },
  ];

  return (
    <div className="w-fit space-y-2 px-4 pt-4">
      {topLevelMenuItems.map((item) => (
        <MenuItem
          key={item.type}
          name={item.name}
          number={item.number}
          selected={defaultState === item.type}
          onClick={() => setDefaultState(item.type as DefaultStateType)}
        />
      ))}
      <div className="ml-[0.5365rem] space-y-2 border-l border-black/10 pl-[0.75rem]">
        {subMenuItems.map((item) => (
          <MenuItem
            key={item.type}
            name={item.name}
            number={item.number}
            className="max-w-[11.5339rem]"
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