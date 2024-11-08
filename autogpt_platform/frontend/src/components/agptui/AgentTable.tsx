// AgentTable.tsx
import * as React from "react";
import { AgentTableRow, AgentTableRowProps } from "./AgentTableRow";
import { AgentTableCard } from "./AgentTableCard";

export interface AgentTableProps {
  agents: AgentTableRowProps[];
}

export const AgentTable: React.FC<AgentTableProps> = ({ agents }) => {
  return (
    <div className="w-full">
      {/* Table header - Hide on mobile */}
      <div className="hidden md:flex flex-col">
        <div className="border-t border-neutral-300" />
        <div className="flex items-center px-4 py-2">
          <div className="flex items-center">
            <div className="flex items-center">
              <input 
                type="checkbox" 
                id="selectAllAgents"
                aria-label="Select all agents"
                className="w-5 h-5 rounded border-2 border-neutral-400 mr-4" 
              />
              <label 
                htmlFor="selectAllAgents" 
                className="text-sm font-medium text-neutral-800"
              >
                Select all
              </label>
            </div>
          </div>
          <div className="grid grid-cols-[400px,150px,150px,100px,100px,50px] w-full items-center ml-4">
            <div className="text-sm font-medium text-neutral-800">Agent info</div>
            <div className="text-sm font-medium text-neutral-800">Date submitted</div>
            <div className="text-sm font-medium text-neutral-800">Status</div>
            <div className="text-sm font-medium text-neutral-800 text-right">Runs</div>
            <div className="text-sm font-medium text-neutral-800 text-right">Reviews</div>
            <div></div>
          </div>
        </div>
        <div className="border-b border-neutral-300" />
      </div>

      {/* Table body */}
      {agents.length > 0 ? (
        <div className="flex flex-col">
          {agents.map((agent, index) => (
            <div key={index} className="md:block">
              <AgentTableRow {...agent} />
              <div className="block md:hidden">
                <AgentTableCard {...agent} />
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="py-4 text-center font-['Geist'] text-base text-neutral-600">
          No agents available. Create your first agent to get started!
        </div>
      )}
    </div>
  );
};
