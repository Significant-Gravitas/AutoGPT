"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { User, Envelope, DownloadSimple } from "@phosphor-icons/react";
import { useGetV2GetWaitlistSignups } from "@/app/api/__generated__/endpoints/admin/admin";

type WaitlistSignupsDialogProps = {
  waitlistId: string;
  onClose: () => void;
};

export function WaitlistSignupsDialog({
  waitlistId,
  onClose,
}: WaitlistSignupsDialogProps) {
  const {
    data: signupsResponse,
    isLoading,
    isError,
  } = useGetV2GetWaitlistSignups(waitlistId);

  const signups = signupsResponse?.status === 200 ? signupsResponse.data : null;

  function exportToCSV() {
    if (!signups) return;

    const headers = ["Type", "Email", "User ID", "Username"];
    const rows = signups.signups.map((signup) => [
      signup.type,
      signup.email || "",
      signup.userId || "",
      signup.username || "",
    ]);

    const escapeCell = (cell: string) => `"${cell.replace(/"/g, '""')}"`;

    const csvContent = [
      headers.join(","),
      ...rows.map((row) => row.map(escapeCell).join(",")),
    ].join("\n");

    const blob = new Blob([csvContent], { type: "text/csv" });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `waitlist-${waitlistId}-signups.csv`;
    a.click();
    window.URL.revokeObjectURL(url);
  }

  function renderContent() {
    if (isLoading) {
      return <div className="py-10 text-center">Loading signups...</div>;
    }

    if (isError) {
      return (
        <div className="py-10 text-center text-red-500">
          Failed to load signups. Please try again.
        </div>
      );
    }

    if (!signups || signups.signups.length === 0) {
      return (
        <div className="py-10 text-center text-gray-500">
          No signups yet for this waitlist.
        </div>
      );
    }

    return (
      <>
        <div className="flex justify-end">
          <Button variant="secondary" size="small" onClick={exportToCSV}>
            <DownloadSimple className="mr-2 h-4 w-4" size={16} />
            Export CSV
          </Button>
        </div>
        <div className="max-h-[400px] overflow-y-auto rounded-md border">
          <table className="w-full">
            <thead className="bg-gray-50 dark:bg-gray-800">
              <tr>
                <th className="px-4 py-3 text-left text-sm font-medium">
                  Type
                </th>
                <th className="px-4 py-3 text-left text-sm font-medium">
                  Email / Username
                </th>
                <th className="px-4 py-3 text-left text-sm font-medium">
                  User ID
                </th>
              </tr>
            </thead>
            <tbody className="divide-y">
              {signups.signups.map((signup, index) => (
                <tr key={index}>
                  <td className="px-4 py-3">
                    {signup.type === "user" ? (
                      <span className="flex items-center gap-1 text-blue-600">
                        <User className="h-4 w-4" size={16} /> User
                      </span>
                    ) : (
                      <span className="flex items-center gap-1 text-gray-600">
                        <Envelope className="h-4 w-4" size={16} /> Email
                      </span>
                    )}
                  </td>
                  <td className="px-4 py-3">
                    {signup.type === "user"
                      ? signup.username || signup.email
                      : signup.email}
                  </td>
                  <td className="px-4 py-3 font-mono text-sm">
                    {signup.userId || "-"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </>
    );
  }

  return (
    <Dialog
      title="Waitlist Signups"
      controlled={{
        isOpen: true,
        set: async (open) => {
          if (!open) onClose();
        },
      }}
      onClose={onClose}
      styling={{ maxWidth: "700px" }}
    >
      <Dialog.Content>
        <p className="mb-4 text-sm text-zinc-500">
          {signups
            ? `${signups.totalCount} total signups`
            : "Loading signups..."}
        </p>

        {renderContent()}

        <Dialog.Footer>
          <Button variant="secondary" onClick={onClose}>
            Close
          </Button>
        </Dialog.Footer>
      </Dialog.Content>
    </Dialog>
  );
}
