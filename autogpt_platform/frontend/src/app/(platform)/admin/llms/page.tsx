import { withRoleAccess } from "@/lib/withRoleAccess";
import { getLlmRegistryPageData } from "./getLlmRegistryPage";
import { LlmRegistryDashboard } from "./components/LlmRegistryDashboard";

async function LlmRegistryPage() {
  const data = await getLlmRegistryPageData();
  return <LlmRegistryDashboard {...data} />;
}

export default async function AdminLlmRegistryPage() {
  const withAdminAccess = await withRoleAccess(["admin"]);
  const ProtectedLlmRegistryPage = await withAdminAccess(LlmRegistryPage);
  return <ProtectedLlmRegistryPage />;
}
