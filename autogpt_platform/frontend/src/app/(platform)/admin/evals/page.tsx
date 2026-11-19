import { withRoleAccess } from "@/lib/withRoleAccess";

import { EvalsContent } from "./components/EvalsContent";

export default async function EvalsPage() {
  "use server";
  const withAdminAccess = await withRoleAccess(["admin"]);
  const ProtectedEvals = await withAdminAccess(EvalsContent);
  return <ProtectedEvals />;
}
