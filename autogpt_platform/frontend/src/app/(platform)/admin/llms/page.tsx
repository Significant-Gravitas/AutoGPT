import { withRoleAccess } from "@/lib/withRoleAccess";
import { LlmRegistryDashboard } from "./components/LlmRegistryDashboard/LlmRegistryDashboard";

function LlmRegistryPage() {
  return (
    <div className="mx-auto p-6">
      <div className="flex flex-col gap-4">
        <div>
          <h1 className="text-3xl font-bold">LLM Registry</h1>
          <p className="text-muted-foreground">
            Models, providers, creators, and copilot routing served from the
            model registry. Read-only for now — editing lands in the next
            release.
          </p>
        </div>
        <LlmRegistryDashboard />
      </div>
    </div>
  );
}

export default async function LlmRegistryAdminPage() {
  const withAdminAccess = await withRoleAccess(["admin"]);
  const ProtectedDashboard = await withAdminAccess(LlmRegistryPage);
  return <ProtectedDashboard />;
}
