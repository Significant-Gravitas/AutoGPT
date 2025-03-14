import { withRoleAccess } from "@/lib/withRoleAccess";
import React from "react";
import { AdminAgentsDataTable } from "@/components/admin/agents/AdminAgentsDataTable";

async function AgentsSettings() {
  return (
    <div className="container mx-auto p-6">
      <div className="flex flex-col gap-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold">Agent Operations</h1>
            <p className="text-gray-500">
              Unified view for agent management and approval history
            </p>
          </div>
        </div>

        <AdminAgentsDataTable />
      </div>
    </div>
  );
}

export default async function AgentSettingsPage() {
  "use server";
  const withAdminAccess = await withRoleAccess(["admin"]);
  const ProtectedAgentSettings = await withAdminAccess(AgentsSettings);
  return <ProtectedAgentSettings />;
}
