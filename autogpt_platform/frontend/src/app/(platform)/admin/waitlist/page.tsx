import { withRoleAccess } from "@/lib/withRoleAccess";
import { Suspense } from "react";
import { WaitlistTable } from "./components/WaitlistTable";
import { CreateWaitlistButton } from "./components/CreateWaitlistButton";

function WaitlistDashboard() {
  return (
    <div className="mx-auto p-6">
      <div className="flex flex-col gap-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold">Waitlist Management</h1>
            <p className="text-gray-500">
              Manage upcoming agent waitlists and track signups
            </p>
          </div>
          <CreateWaitlistButton />
        </div>

        <Suspense
          fallback={
            <div className="py-10 text-center">Loading waitlists...</div>
          }
        >
          <WaitlistTable />
        </Suspense>
      </div>
    </div>
  );
}

export default async function WaitlistDashboardPage() {
  "use server";
  const withAdminAccess = await withRoleAccess(["admin"]);
  const ProtectedWaitlistDashboard = await withAdminAccess(WaitlistDashboard);
  return <ProtectedWaitlistDashboard />;
}
