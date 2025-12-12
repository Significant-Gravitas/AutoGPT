import { withRoleAccess } from "@/lib/withRoleAccess";
import { useLlmRegistryPage } from "./useLlmRegistryPage";
import { LlmRegistryDashboard } from "./components/LlmRegistryDashboard";

async function LlmRegistryPage() {
  const data = await useLlmRegistryPage();
  return <LlmRegistryDashboard {...data} />;
}

export default async function AdminLlmRegistryPage() {
  "use server";
  const withAdminAccess = await withRoleAccess(["admin"]);
  const ProtectedLlmRegistryPage = await withAdminAccess(LlmRegistryPage);
  return <ProtectedLlmRegistryPage />;
}
