"use client";

import { useState } from "react";
import {
  TableRow,
  TableCell,
  Table,
  TableHeader,
  TableHead,
  TableBody,
} from "@/components/__legacy__/ui/table";
import { Badge } from "@/components/__legacy__/ui/badge";
import { ChevronDown, ChevronRight } from "lucide-react";
import { formatDistanceToNow } from "date-fns";
import type { StoreListingWithVersionsAdminView } from "@/app/api/__generated__/models/storeListingWithVersionsAdminView";
import type { StoreSubmissionAdminView } from "@/app/api/__generated__/models/storeSubmissionAdminView";
import { SubmissionStatus } from "@/app/api/__generated__/models/submissionStatus";
import { ApproveRejectButtons } from "./ApproveRejectButton";
import { DownloadAgentAdminButton } from "./DownloadAgentButton";

// Moved the getStatusBadge function into the client component
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

export function ExpandableRow({
  listing,
  latestVersion,
}: {
  listing: StoreListingWithVersionsAdminView;
  latestVersion: StoreSubmissionAdminView | null;
}) {
  const [expanded, setExpanded] = useState(false);

  return (
    <>
      <TableRow className="cursor-pointer hover:bg-muted/50">
        <TableCell onClick={() => setExpanded(!expanded)}>
          {expanded ? (
            <ChevronDown className="h-4 w-4" />
          ) : (
            <ChevronRight className="h-4 w-4" />
          )}
        </TableCell>
        <TableCell
          className="font-medium"
          onClick={() => setExpanded(!expanded)}
        >
          {latestVersion?.name || "Unnamed Agent"}
        </TableCell>
        <TableCell onClick={() => setExpanded(!expanded)}>
          {listing.creator_email || "Unknown"}
        </TableCell>
        <TableCell onClick={() => setExpanded(!expanded)}>
          {latestVersion?.sub_heading || "No description"}
        </TableCell>
        <TableCell onClick={() => setExpanded(!expanded)}>
          {latestVersion?.status && getStatusBadge(latestVersion.status)}
        </TableCell>
        <TableCell onClick={() => setExpanded(!expanded)}>
          {latestVersion?.submitted_at
            ? formatDistanceToNow(new Date(latestVersion.submitted_at), {
                addSuffix: true,
              })
            : "Unknown"}
        </TableCell>
        <TableCell className="text-right">
          <div className="flex justify-end gap-2">
            {latestVersion?.listing_version_id && (
              <DownloadAgentAdminButton
                storeListingVersionId={latestVersion.listing_version_id}
              />
            )}

            {(latestVersion?.status === SubmissionStatus.PENDING ||
              latestVersion?.status === SubmissionStatus.APPROVED) && (
              <ApproveRejectButtons version={latestVersion} />
            )}
          </div>
        </TableCell>
      </TableRow>

      {/* Expanded version history */}
      {expanded && (
        <TableRow>
          <TableCell colSpan={7} className="border-t-0 p-0">
            <div className="bg-muted/30 px-4 py-3">
              <h4 className="mb-2 text-sm font-semibold">Version History</h4>
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
                    <TableHead>Name</TableHead>
                    <TableHead>Sub Heading</TableHead>
                    <TableHead>Description</TableHead>
                    {/* <TableHead>Categories</TableHead> */}
                    <TableHead className="text-right">Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {(listing.versions ?? [])
                    .sort(
                      (a, b) =>
                        (b.listing_version ?? 1) - (a.listing_version ?? 0),
                    )
                    .map((version) => (
                      <TableRow key={version.listing_version_id}>
                        <TableCell>
                          v{version.listing_version || "?"}
                          {version.listing_version_id ===
                            listing.active_listing_version_id && (
                            <Badge className="ml-2 bg-blue-500">Active</Badge>
                          )}
                        </TableCell>
                        <TableCell>{getStatusBadge(version.status)}</TableCell>
                        <TableCell>
                          {version.changes_summary || "No summary"}
                        </TableCell>
                        <TableCell>
                          {version.submitted_at
                            ? formatDistanceToNow(
                                new Date(version.submitted_at),
                                { addSuffix: true },
                              )
                            : "Unknown"}
                        </TableCell>
                        <TableCell>
                          {version.reviewed_at
                            ? formatDistanceToNow(
                                new Date(version.reviewed_at),
                                {
                                  addSuffix: true,
                                },
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
                            <span className="text-gray-400">
                              No external comments
                            </span>
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
                            <span className="text-gray-400">
                              No internal comments
                            </span>
                          )}
                        </TableCell>
                        <TableCell>{version.name}</TableCell>
                        <TableCell>{version.sub_heading}</TableCell>
                        <TableCell>{version.description}</TableCell>
                        {/* <TableCell>{version.categories.join(", ")}</TableCell> */}
                        <TableCell className="text-right">
                          <div className="flex justify-end gap-2">
                            {version.listing_version_id && (
                              <DownloadAgentAdminButton
                                storeListingVersionId={
                                  version.listing_version_id
                                }
                              />
                            )}
                            {(version.status === SubmissionStatus.PENDING ||
                              version.status === SubmissionStatus.APPROVED) && (
                              <ApproveRejectButtons version={version} />
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
  );
}
