"use client";

import { useEffect, useState } from "react";
import { Button } from "@/components/atoms/Button/Button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/__legacy__/ui/dialog";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/__legacy__/ui/table";
import {
  getWaitlistSignups,
  type WaitlistSignup,
  type WaitlistSignupListResponse,
} from "../actions";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { User, Mail, Download } from "lucide-react";

type WaitlistSignupsDialogProps = {
  waitlistId: string;
  onClose: () => void;
};

export function WaitlistSignupsDialog({
  waitlistId,
  onClose,
}: WaitlistSignupsDialogProps) {
  const [loading, setLoading] = useState(true);
  const [signups, setSignups] = useState<WaitlistSignupListResponse | null>(
    null,
  );
  const { toast } = useToast();

  useEffect(() => {
    async function loadSignups() {
      try {
        const response = await getWaitlistSignups(waitlistId);
        setSignups(response);
      } catch (error) {
        console.error("Error loading signups:", error);
        toast({
          variant: "destructive",
          title: "Error",
          description: "Failed to load signups",
        });
      } finally {
        setLoading(false);
      }
    }

    loadSignups();
  }, [waitlistId, toast]);

  function exportToCSV() {
    if (!signups) return;

    const headers = ["Type", "Email", "User ID", "Username"];
    const rows = signups.signups.map((signup) => [
      signup.type,
      signup.email || "",
      signup.userId || "",
      signup.username || "",
    ]);

    const csvContent = [
      headers.join(","),
      ...rows.map((row) => row.map((cell) => `"${cell}"`).join(",")),
    ].join("\n");

    const blob = new Blob([csvContent], { type: "text/csv" });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `waitlist-${waitlistId}-signups.csv`;
    a.click();
    window.URL.revokeObjectURL(url);
  }

  return (
    <Dialog open={true} onOpenChange={onClose}>
      <DialogContent className="max-h-[90vh] overflow-y-auto sm:max-w-[700px]">
        <DialogHeader>
          <DialogTitle>Waitlist Signups</DialogTitle>
          <DialogDescription>
            {signups
              ? `${signups.totalCount} total signups`
              : "Loading signups..."}
          </DialogDescription>
        </DialogHeader>

        {loading ? (
          <div className="py-10 text-center">Loading signups...</div>
        ) : signups && signups.signups.length > 0 ? (
          <>
            <div className="flex justify-end">
              <Button variant="secondary" size="small" onClick={exportToCSV}>
                <Download className="mr-2 h-4 w-4" />
                Export CSV
              </Button>
            </div>
            <div className="max-h-[400px] overflow-y-auto rounded-md border">
              <Table>
                <TableHeader className="bg-gray-50">
                  <TableRow>
                    <TableHead className="font-medium">Type</TableHead>
                    <TableHead className="font-medium">
                      Email / Username
                    </TableHead>
                    <TableHead className="font-medium">User ID</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {signups.signups.map((signup, index) => (
                    <TableRow key={index}>
                      <TableCell>
                        {signup.type === "user" ? (
                          <span className="flex items-center gap-1 text-blue-600">
                            <User className="h-4 w-4" /> User
                          </span>
                        ) : (
                          <span className="flex items-center gap-1 text-gray-600">
                            <Mail className="h-4 w-4" /> Email
                          </span>
                        )}
                      </TableCell>
                      <TableCell>
                        {signup.type === "user"
                          ? signup.username || signup.email
                          : signup.email}
                      </TableCell>
                      <TableCell className="font-mono text-sm">
                        {signup.userId || "-"}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          </>
        ) : (
          <div className="py-10 text-center text-gray-500">
            No signups yet for this waitlist.
          </div>
        )}

        <div className="flex justify-end">
          <Button variant="secondary" onClick={onClose}>
            Close
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}
