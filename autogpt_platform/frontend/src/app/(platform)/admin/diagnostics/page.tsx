import { withRoleAccess } from "@/lib/withRoleAccess";
import { DiagnosticsContent } from "./components/DiagnosticsContent";

function AdminDiagnostics() {
  return (
    <div className="mx-auto p-6">
      <DiagnosticsContent />
    </div>
  );
}

export default async function AdminDiagnosticsPage() {
  "use server";
  const withAdminAccess = await withRoleAccess(["admin"]);
  const ProtectedAdminDiagnostics = await withAdminAccess(AdminDiagnostics);
  return <ProtectedAdminDiagnostics />;
}
