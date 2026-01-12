"use client";

import { useEffect, useState } from "react";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/__legacy__/ui/table";
import { Button } from "@/components/atoms/Button/Button";
import { getWaitlistsAdmin, deleteWaitlist } from "../actions";
import type { WaitlistAdminResponse } from "@/lib/autogpt-server-api/types";
import { EditWaitlistDialog } from "./EditWaitlistDialog";
import { WaitlistSignupsDialog } from "./WaitlistSignupsDialog";
import { Trash, PencilSimple, Users, Link } from "@phosphor-icons/react";
import { useToast } from "@/components/molecules/Toast/use-toast";

export function WaitlistTable() {
  const [waitlists, setWaitlists] = useState<WaitlistAdminResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [editingWaitlist, setEditingWaitlist] =
    useState<WaitlistAdminResponse | null>(null);
  const [viewingSignups, setViewingSignups] = useState<string | null>(null);
  const { toast } = useToast();

  async function loadWaitlists() {
    try {
      const response = await getWaitlistsAdmin();
      setWaitlists(response.waitlists);
    } catch (error) {
      console.error("Error loading waitlists:", error);
      toast({
        variant: "destructive",
        title: "Error",
        description: "Failed to load waitlists",
      });
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadWaitlists();
  }, []);

  async function handleDelete(waitlistId: string) {
    if (!confirm("Are you sure you want to delete this waitlist?")) return;

    try {
      await deleteWaitlist(waitlistId);
      toast({
        title: "Success",
        description: "Waitlist deleted successfully",
      });
      loadWaitlists();
    } catch (error) {
      console.error("Error deleting waitlist:", error);
      toast({
        variant: "destructive",
        title: "Error",
        description: "Failed to delete waitlist",
      });
    }
  }

  function formatStatus(status: string) {
    const statusColors: Record<string, string> = {
      NOT_STARTED: "bg-gray-100 text-gray-800",
      WORK_IN_PROGRESS: "bg-blue-100 text-blue-800",
      DONE: "bg-green-100 text-green-800",
      CANCELED: "bg-red-100 text-red-800",
    };

    return (
      <span
        className={`rounded-full px-2 py-1 text-xs font-medium ${statusColors[status] || "bg-gray-100"}`}
      >
        {status.replace(/_/g, " ")}
      </span>
    );
  }

  function formatDate(dateStr: string) {
    if (!dateStr) return "-";
    return new Intl.DateTimeFormat("en-US", {
      month: "short",
      day: "numeric",
      year: "numeric",
    }).format(new Date(dateStr));
  }

  if (loading) {
    return <div className="py-10 text-center">Loading waitlists...</div>;
  }

  if (waitlists.length === 0) {
    return (
      <div className="py-10 text-center text-gray-500">
        No waitlists found. Create one to get started!
      </div>
    );
  }

  return (
    <>
      <div className="rounded-md border bg-white">
        <Table>
          <TableHeader className="bg-gray-50">
            <TableRow>
              <TableHead className="font-medium">Name</TableHead>
              <TableHead className="font-medium">Status</TableHead>
              <TableHead className="font-medium">Signups</TableHead>
              <TableHead className="font-medium">Votes</TableHead>
              <TableHead className="font-medium">Created</TableHead>
              <TableHead className="font-medium">Linked Agent</TableHead>
              <TableHead className="font-medium">Actions</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {waitlists.map((waitlist) => (
              <TableRow key={waitlist.id}>
                <TableCell>
                  <div>
                    <div className="font-medium">{waitlist.name}</div>
                    <div className="text-sm text-gray-500">
                      {waitlist.subHeading}
                    </div>
                  </div>
                </TableCell>
                <TableCell>{formatStatus(waitlist.status)}</TableCell>
                <TableCell>{waitlist.signupCount}</TableCell>
                <TableCell>{waitlist.votes}</TableCell>
                <TableCell>{formatDate(waitlist.createdAt)}</TableCell>
                <TableCell>
                  {waitlist.storeListingId ? (
                    <span className="text-green-600">
                      <Link size={16} className="inline" /> Linked
                    </span>
                  ) : (
                    <span className="text-gray-400">Not linked</span>
                  )}
                </TableCell>
                <TableCell>
                  <div className="flex gap-2">
                    <Button
                      variant="ghost"
                      size="small"
                      onClick={() => setViewingSignups(waitlist.id)}
                      title="View signups"
                    >
                      <Users size={16} />
                    </Button>
                    <Button
                      variant="ghost"
                      size="small"
                      onClick={() => setEditingWaitlist(waitlist)}
                      title="Edit"
                    >
                      <PencilSimple size={16} />
                    </Button>
                    <Button
                      variant="ghost"
                      size="small"
                      onClick={() => handleDelete(waitlist.id)}
                      title="Delete"
                    >
                      <Trash size={16} className="text-red-500" />
                    </Button>
                  </div>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>

      {editingWaitlist && (
        <EditWaitlistDialog
          waitlist={editingWaitlist}
          onClose={() => setEditingWaitlist(null)}
          onSave={() => {
            setEditingWaitlist(null);
            loadWaitlists();
          }}
        />
      )}

      {viewingSignups && (
        <WaitlistSignupsDialog
          waitlistId={viewingSignups}
          onClose={() => setViewingSignups(null)}
        />
      )}
    </>
  );
}
