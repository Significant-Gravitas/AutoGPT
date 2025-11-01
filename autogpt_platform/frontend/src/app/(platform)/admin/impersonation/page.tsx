import { AdminImpersonationPanel } from "@/components/admin/AdminImpersonationPanel";

export default function AdminImpersonationPage() {
  return (
    <div className="container mx-auto space-y-6 py-6">
      <div className="space-y-2">
        <h1 className="text-3xl font-bold tracking-tight">
          User Impersonation
        </h1>
        <p className="text-gray-600">
          Manage admin user impersonation for debugging and support purposes
        </p>
      </div>

      <AdminImpersonationPanel />
    </div>
  );
}
