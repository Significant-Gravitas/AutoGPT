"use client";

import { useState, useEffect } from "react";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  ChevronDown,
  ChevronRight,
  Search,
  CheckCircle,
  XCircle,
  ExternalLink
} from "lucide-react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import {
  StoreListingWithVersions,
  StoreSubmission,
  SubmissionStatus,
} from "@/lib/autogpt-server-api/types";
import {
  getAdminListingsWithVersions,
  approveAgent,
  rejectAgent
} from "@/app/admin/agents/actions";
import { formatDistanceToNow } from "date-fns";

export function AdminAgentsDataTable() {
  const [listings, setListings] = useState<StoreListingWithVersions[]>([]);
  const [loading, setLoading] = useState(true);
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedStatus, setSelectedStatus] = useState<SubmissionStatus | null>(
    SubmissionStatus.PENDING
  );
  const [expandedListings, setExpandedListings] = useState<
    Record<string, boolean>
  >({});

  // Dialog state for approve/reject functionality
  const [selectedVersion, setSelectedVersion] = useState<StoreSubmission | null>(null);
  const [isApproveDialogOpen, setIsApproveDialogOpen] = useState(false);
  const [isRejectDialogOpen, setIsRejectDialogOpen] = useState(false);

  useEffect(() => {
    fetchListings();
  }, [currentPage, selectedStatus]);

  const fetchListings = async () => {
    setLoading(true);
    try {
      const response = await getAdminListingsWithVersions(
        selectedStatus || undefined,
        searchQuery || undefined,
        currentPage,
        10
      );

      setListings(response.listings);
      setTotalPages(response.pagination.total_pages);
    } catch (error) {
      console.error("Error fetching listings:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleSearch = () => {
    setCurrentPage(1);
    fetchListings();
  };

  const handleStatusChange = (status: string) => {
    if (status === "ALL") {
      setSelectedStatus(null);
    } else {
      setSelectedStatus(status as SubmissionStatus);
    }
    setCurrentPage(1);
  };

  const toggleListingExpanded = (listingId: string) => {
    setExpandedListings((prev) => ({
      ...prev,
      [listingId]: !prev[listingId],
    }));
  };

  const handleApproveClick = (version: StoreSubmission) => {
    setSelectedVersion(version);
    setIsApproveDialogOpen(true);
  };

  const handleRejectClick = (version: StoreSubmission) => {
    setSelectedVersion(version);
    setIsRejectDialogOpen(true);
  };

  const handleApproveSubmit = async (formData: FormData) => {
    setIsApproveDialogOpen(false);
    try {
      await approveAgent(formData);
      fetchListings(); // Refresh data after approval
    } catch (error) {
      console.error("Error approving agent:", error);
    }
  };

  const handleRejectSubmit = async (formData: FormData) => {
    setIsRejectDialogOpen(false);
    try {
      await rejectAgent(formData);
      fetchListings(); // Refresh data after rejection
    } catch (error) {
      console.error("Error rejecting agent:", error);
    }
  };

  const getStatusBadge = (status: SubmissionStatus) => {
    switch (status) {
      case SubmissionStatus.PENDING:
        return <Badge className="bg-amber-500">Pending</Badge>;
      case SubmissionStatus.APPROVED:
        return <Badge className="bg-green-500">Approved</Badge>;
      case SubmissionStatus.REJECTED:
        return <Badge className="bg-red-500">Rejected</Badge>;
      default:
        return <Badge className="bg-gray-500">Draft</Badge>;
    }
  };

  const Pagination = ({
    currentPage,
    totalPages,
    onPageChange,
  }: {
    currentPage: number;
    totalPages: number;
    onPageChange: (page: number) => void;
  }) => (
    <div className="mt-4 flex items-center justify-center space-x-2">
      <Button
        variant="outline"
        onClick={() => onPageChange(currentPage - 1)}
        disabled={currentPage <= 1}
      >
        Previous
      </Button>
      <span className="text-sm">
        Page {currentPage} of {totalPages}
      </span>
      <Button
        variant="outline"
        onClick={() => onPageChange(currentPage + 1)}
        disabled={currentPage >= totalPages}
      >
        Next
      </Button>
    </div>
  );

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex w-full max-w-sm items-center gap-2">
          <Input
            placeholder="Search submissions..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSearch()}
          />
          <Button variant="outline" onClick={handleSearch}>
            <Search className="h-4 w-4" />
          </Button>
        </div>

        <Select
          value={selectedStatus !== null ? selectedStatus : "ALL"}
          onValueChange={handleStatusChange}
        >
          <SelectTrigger className="w-[180px]">
            <SelectValue placeholder="Select Status" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="ALL">All</SelectItem>
            <SelectItem value={SubmissionStatus.PENDING}>Pending</SelectItem>
            <SelectItem value={SubmissionStatus.APPROVED}>Approved</SelectItem>
            <SelectItem value={SubmissionStatus.REJECTED}>Rejected</SelectItem>
          </SelectContent>
        </Select>
      </div>

      <div className="rounded-md border">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="w-10"></TableHead>
              <TableHead>Name</TableHead>
              <TableHead>Creator</TableHead>
              <TableHead>Description</TableHead>
              <TableHead>Status</TableHead>
              <TableHead>Submitted</TableHead>
              <TableHead className="text-right">Actions</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {loading ? (
              <TableRow>
                <TableCell colSpan={7} className="py-10 text-center">
                  Loading...
                </TableCell>
              </TableRow>
            ) : listings.length === 0 ? (
              <TableRow>
                <TableCell colSpan={7} className="py-10 text-center">
                  No submissions found
                </TableCell>
              </TableRow>
            ) : (
              listings.map((listing) => (
                <>
                  <TableRow
                    key={listing.listing_id}
                    className="cursor-pointer hover:bg-muted/50"
                  >
                    <TableCell
                      onClick={() => toggleListingExpanded(listing.listing_id)}
                    >
                      {expandedListings[listing.listing_id] ? (
                        <ChevronDown className="h-4 w-4" />
                      ) : (
                        <ChevronRight className="h-4 w-4" />
                      )}
                    </TableCell>
                    <TableCell
                      className="font-medium"
                      onClick={() => toggleListingExpanded(listing.listing_id)}
                    >
                      {listing.latest_version?.name || "Unnamed Agent"}
                    </TableCell>
                    <TableCell
                      onClick={() => toggleListingExpanded(listing.listing_id)}
                    >
                      {listing.creator_email || "Unknown"}
                    </TableCell>
                    <TableCell
                      onClick={() => toggleListingExpanded(listing.listing_id)}
                    >
                      {listing.latest_version?.sub_heading || "No description"}
                    </TableCell>
                    <TableCell
                      onClick={() => toggleListingExpanded(listing.listing_id)}
                    >
                      {listing.latest_version?.status &&
                        getStatusBadge(listing.latest_version.status)}
                    </TableCell>
                    <TableCell
                      onClick={() => toggleListingExpanded(listing.listing_id)}
                    >
                      {listing.latest_version?.date_submitted
                        ? formatDistanceToNow(
                          new Date(listing.latest_version.date_submitted),
                          { addSuffix: true },
                        )
                        : "Unknown"}
                    </TableCell>
                    <TableCell className="text-right">
                      <div className="flex justify-end gap-2">
                        <Button size="sm" variant="outline">
                          <ExternalLink className="h-4 w-4 mr-2" />
                          Builder
                        </Button>

                        {listing.latest_version?.status === SubmissionStatus.PENDING && (
                          <>
                            <Button
                              size="sm"
                              variant="outline"
                              className="text-green-600 hover:text-green-700 hover:bg-green-50"
                              onClick={(e) => {
                                e.stopPropagation();
                                handleApproveClick(listing.latest_version!);
                              }}
                            >
                              <CheckCircle className="h-4 w-4 mr-2" />
                              Approve
                            </Button>
                            <Button
                              size="sm"
                              variant="outline"
                              className="text-red-600 hover:text-red-700 hover:bg-red-50"
                              onClick={(e) => {
                                e.stopPropagation();
                                handleRejectClick(listing.latest_version!);
                              }}
                            >
                              <XCircle className="h-4 w-4 mr-2" />
                              Reject
                            </Button>
                          </>
                        )}
                      </div>
                    </TableCell>
                  </TableRow>

                  {/* Expanded version history */}
                  {expandedListings[listing.listing_id] && (
                    <TableRow>
                      <TableCell colSpan={7} className="border-t-0 p-0">
                        <div className="bg-muted/30 px-4 py-3">
                          <h4 className="mb-2 text-sm font-semibold">
                            Version History
                          </h4>
                          <Table>
                            <TableHeader>
                              <TableRow>
                                <TableHead>Version</TableHead>
                                <TableHead>Status</TableHead>
                                <TableHead>Changes</TableHead>
                                <TableHead>Submitted</TableHead>
                                <TableHead>Reviewed</TableHead>
                                <TableHead>External Comments</TableHead>
                                <TableHead>Internal Comments</TableHead>
                                <TableHead className="text-right">
                                  Actions
                                </TableHead>
                              </TableRow>
                            </TableHeader>
                            <TableBody>
                              {listing.versions.map((version) => (
                                <TableRow
                                  key={version.store_listing_version_id}
                                >
                                  <TableCell>
                                    v{version.version || "?"}
                                    {version.store_listing_version_id ===
                                      listing.active_version_id && (
                                        <Badge className="ml-2 bg-blue-500">
                                          Active
                                        </Badge>
                                      )}
                                  </TableCell>
                                  <TableCell>
                                    {getStatusBadge(version.status)}
                                  </TableCell>
                                  <TableCell>
                                    {version.changes_summary || "No summary"}
                                  </TableCell>
                                  <TableCell>
                                    {version.date_submitted
                                      ? formatDistanceToNow(
                                        new Date(version.date_submitted),
                                        { addSuffix: true },
                                      )
                                      : "Unknown"}
                                  </TableCell>
                                  <TableCell>
                                    {version.reviewed_at
                                      ? formatDistanceToNow(
                                        new Date(version.reviewed_at),
                                        { addSuffix: true },
                                      )
                                      : "Not reviewed"}
                                  </TableCell>
                                  <TableCell className="max-w-xs truncate">
                                    {version.review_comments ? (
                                      <div
                                        className="truncate"
                                        title={version.review_comments}
                                      >
                                        {version.review_comments}
                                      </div>
                                    ) : (
                                      <span className="text-gray-400">No external comments</span>
                                    )}
                                  </TableCell>
                                  <TableCell className="max-w-xs truncate">
                                    {version.internal_comments ? (
                                      <div
                                        className="truncate text-pink-600"
                                        title={version.internal_comments}
                                      >
                                        {version.internal_comments}
                                      </div>
                                    ) : (
                                      <span className="text-gray-400">No internal comments</span>
                                    )}
                                  </TableCell>
                                  <TableCell className="text-right">
                                    <div className="flex justify-end gap-2">
                                      <Button
                                        size="sm"
                                        variant="outline"
                                        onClick={() => window.location.href = `/admin/agents/${version.store_listing_version_id}`}
                                      >
                                        Review
                                      </Button>

                                      {version.status === SubmissionStatus.PENDING && (
                                        <>
                                          <Button
                                            size="sm"
                                            variant="outline"
                                            className="text-green-600 hover:text-green-700 hover:bg-green-50"
                                            onClick={() => handleApproveClick(version)}
                                          >
                                            <CheckCircle className="h-4 w-4 mr-1" />
                                            Approve
                                          </Button>
                                          <Button
                                            size="sm"
                                            variant="outline"
                                            className="text-red-600 hover:text-red-700 hover:bg-red-50"
                                            onClick={() => handleRejectClick(version)}
                                          >
                                            <XCircle className="h-4 w-4 mr-1" />
                                            Reject
                                          </Button>
                                        </>
                                      )}
                                    </div>
                                  </TableCell>
                                </TableRow>
                              ))}
                            </TableBody>
                          </Table>
                        </div>
                      </TableCell>
                    </TableRow>
                  )}
                </>
              ))
            )}
          </TableBody>
        </Table>
      </div>

      <Pagination
        currentPage={currentPage}
        totalPages={totalPages}
        onPageChange={setCurrentPage}
      />

      {/* Approve Dialog */}
      <Dialog open={isApproveDialogOpen} onOpenChange={setIsApproveDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Approve Agent</DialogTitle>
            <DialogDescription>
              Are you sure you want to approve this agent? This will make it available in the marketplace.
            </DialogDescription>
          </DialogHeader>

          <form action={handleApproveSubmit}>
            <input
              type="hidden"
              name="id"
              value={selectedVersion?.store_listing_version_id || ""}
            />

            <div className="grid gap-4 py-4">
              <div className="grid gap-2">
                <Label htmlFor="comments">Comments (Optional)</Label>
                <Textarea
                  id="comments"
                  name="comments"
                  placeholder="Add any comments for the agent creator"
                />
              </div>
            </div>

            <DialogFooter>
              <Button
                type="button"
                variant="outline"
                onClick={() => setIsApproveDialogOpen(false)}
              >
                Cancel
              </Button>
              <Button
                type="submit"
              >
                Approve
              </Button>
            </DialogFooter>
          </form>
        </DialogContent>
      </Dialog>

      {/* Reject Dialog */}
      <Dialog open={isRejectDialogOpen} onOpenChange={setIsRejectDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Reject Agent</DialogTitle>
            <DialogDescription>
              Please provide feedback on why this agent is being rejected.
            </DialogDescription>
          </DialogHeader>

          <form action={handleRejectSubmit}>
            <input
              type="hidden"
              name="id"
              value={selectedVersion?.store_listing_version_id || ""}
            />

            <div className="grid gap-4 py-4">
              <div className="grid gap-2">
                <Label htmlFor="comments">Comments for Creator</Label>
                <Textarea
                  id="comments"
                  name="comments"
                  placeholder="Provide feedback for the agent creator"
                  required
                />
              </div>

              <div className="grid gap-2">
                <Label htmlFor="internal_comments">Internal Comments</Label>
                <Textarea
                  id="internal_comments"
                  name="internal_comments"
                  placeholder="Add any internal notes (not visible to creator)"
                />
              </div>
            </div>

            <DialogFooter>
              <Button
                type="button"
                variant="outline"
                onClick={() => setIsRejectDialogOpen(false)}
              >
                Cancel
              </Button>
              <Button
                type="submit"
                variant="destructive"
              >
                Reject
              </Button>
            </DialogFooter>
          </form>
        </DialogContent>
      </Dialog>
    </div>
  );
}