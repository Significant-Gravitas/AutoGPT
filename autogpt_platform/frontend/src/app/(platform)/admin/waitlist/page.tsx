import { withRoleAccess } from "@/lib/withRoleAccess";
import { Suspense } from "react";
import { WaitlistTable } from "./components/WaitlistTable";
import { CreateWaitlistButton } from "./components/CreateWaitlistButton";
import { Warning } from "@phosphor-icons/react/dist/ssr";

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

        <div className="flex items-start gap-3 rounded-lg border border-amber-300 bg-amber-50 p-4 dark:border-amber-700 dark:bg-amber-950">
          <Warning
            className="mt-0.5 h-5 w-5 flex-shrink-0 text-amber-600 dark:text-amber-400"
            weight="fill"
          />
          <div className="text-sm text-amber-800 dark:text-amber-200">
            <p className="font-medium">TODO: Email-only signup notifications</p>
            <p className="mt-1 text-amber-700 dark:text-amber-300">
              Notifications for email-only signups (users who weren&apos;t
              logged in) have not been implemented yet. Currently only
              registered users will receive launch emails.
            </p>
          </div>
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
  const withAdminAccess = await withRoleAccess(["admin"]);
  const ProtectedWaitlistDashboard = await withAdminAccess(WaitlistDashboard);
  return <ProtectedWaitlistDashboard />;
}
