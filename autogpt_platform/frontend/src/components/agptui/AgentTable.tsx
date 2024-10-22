// AgentTable.tsx
import * as React from "react";
import { AgentTableRow, AgentTableRowProps } from "./AgentTableRow";
import { AgentTableCard } from "./AgentTableCard";

export interface AgentTableProps {
  agents: AgentTableRowProps[];
}

export const AgentTable: React.FC<AgentTableProps> = ({ agents }) => {
  return (
    <div className="mx-auto w-full max-w-[1095px]">
      {/* Table for medium and larger screens */}
      <div className="hidden md:block">
        <table className="w-full">
          <thead>
            <tr className="border-b border-[#d9d9d9]">
              <th className="font-['PP Neue Montreal TT'] py-4 text-left text-base leading-[21px] tracking-tight text-[#282828]">
                Agent
              </th>
              <th className="font-['PP Neue Montreal TT'] py-4 text-left text-base leading-[21px] tracking-tight text-[#282828]">
                Date submitted
              </th>
              <th className="font-['PP Neue Montreal TT'] py-4 text-left text-base leading-[21px] tracking-tight text-[#282828]">
                Status
              </th>
              <th className="font-['PP Neue Montreal TT'] py-4 text-left text-base leading-[21px] tracking-tight text-[#282828]">
                Runs
              </th>
              <th className="font-['PP Neue Montreal TT'] py-4 text-left text-base leading-[21px] tracking-tight text-[#282828]">
                Reviews
              </th>
              <th className="font-['PP Neue Montreal TT'] py-4 text-left text-base leading-[21px] tracking-tight text-[#282828]">
                Actions
              </th>
            </tr>
          </thead>
          <tbody>
            {agents.length > 0 ? (
              agents.map((agent, index) => (
                <AgentTableRow key={index} {...agent} />
              ))
            ) : (
              <tr>
                <td
                  colSpan={6}
                  className="font-['PP Neue Montreal TT'] py-4 text-center text-base text-[#282828]"
                >
                  No agents available. Create your first agent to get started!
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
      {/* Cards for small screens */}
      <div className="block md:hidden">
        {agents.length > 0 ? (
          agents.map((agent, index) => (
            <AgentTableCard key={index} {...agent} />
          ))
        ) : (
          <div className="font-['PP Neue Montreal TT'] py-4 text-center text-base text-[#707070]">
            No agents available. Create your first agent to get started!
          </div>
        )}
      </div>
    </div>
  );
};
