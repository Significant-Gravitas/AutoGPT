"use client";

import { useState } from "react";
import {
  TableRow,
  TableCell,
  Table,
  TableHeader,
  TableHead,
  TableBody,
} from "@/components/molecules/Table/components/TablePrimitives";
import { ChevronDown, ChevronRight, Eye } from "lucide-react";
import Link from "next/link";
import { formatDistanceToNow } from "date-fns";
import type { StoreListingWithVersionsAdminView } from "@/app/api/__generated__/models/storeListingWithVersionsAdminView";
import type { StoreSubmissionAdminView } from "@/app/api/__generated__/models/storeSubmissionAdminView";
import { SubmissionStatus } from "@/app/api/__generated__/models/submissionStatus";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { ApproveRejectButtons } from "./ApproveRejectButton";
import { DownloadAgentAdminButton } from "./DownloadAgentButton";

const STATUS_CHIP =
  "inline-flex cursor-default items-center rounded-md border border-transparent px-2.5 py-0.5 text-caption font-semibold leading-4 text-action-foreground shadow";

// Moved the getStatusBadge function into the client component
const getStatusBadge = (status: SubmissionStatus) => {
  switch (status) {
    case SubmissionStatus.PENDING:
      return <span className={cn(STATUS_CHIP, "bg-warning")}>Pending</span>;
    case SubmissionStatus.APPROVED:
      return <span className={cn(STATUS_CHIP, "bg-success")}>Approved</span>;
    case SubmissionStatus.REJECTED:
      return <span className={cn(STATUS_CHIP, "bg-error")}>Rejected</span>;
    default:
      return (
        <span className={cn(STATUS_CHIP, "bg-foreground-tertiary")}>Draft</span>
      );
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
      <TableRow className="cursor-pointer">
        <TableCell onClick={() => setExpanded(!expanded)}>
          {expanded ? (
            <ChevronDown className="h-4 w-4" />
          ) : (
            <ChevronRight className="h-4 w-4" />
          )}
        </TableCell>
        <TableCell onClick={() => setExpanded(!expanded)}>
          <span className="font-medium">
            {latestVersion?.name || "Unnamed Agent"}
          </span>
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
              <>
                <Link
                  href={`/admin/marketplace/preview/${latestVersion.listing_version_id}`}
                >
                  <Button size="xs" variant="secondary">
                    <Eye className="h-4 w-4" />
                    Preview
                  </Button>
                </Link>
                <DownloadAgentAdminButton
                  storeListingVersionId={latestVersion.listing_version_id}
                />
              </>
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
          <td colSpan={7} className="p-0">
            <div className="bg-muted/30 px-4 py-3">
              <Text
                variant="body-semibold-tight"
                as="h4"
                tone="inherit"
                className="mb-2"
              >
                Version History
              </Text>
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
                    <TableHead>Categories</TableHead>
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
                            <span className={cn(STATUS_CHIP, "ml-2 bg-info")}>
                              Active
                            </span>
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
                        <TableCell className="max-w-xs overflow-hidden">
                          {version.review_comments ? (
                            <div
                              className="truncate"
                              title={version.review_comments}
                            >
                              {version.review_comments}
                            </div>
                          ) : (
                            <span className="block truncate text-foreground-subtle">
                              No external comments
                            </span>
                          )}
                        </TableCell>
                        <TableCell className="max-w-xs overflow-hidden">
                          {version.internal_comments ? (
                            <div
                              // oxlint-disable-next-line design-system/no-raw-colors -- pink has no role; a role would change the hue (deferred)
                              className="truncate text-pink-600"
                              title={version.internal_comments}
                            >
                              {version.internal_comments}
                            </div>
                          ) : (
                            <span className="block truncate text-foreground-subtle">
                              No internal comments
                            </span>
                          )}
                        </TableCell>
                        <TableCell>{version.name}</TableCell>
                        <TableCell>{version.sub_heading}</TableCell>
                        <TableCell>{version.description}</TableCell>
                        <TableCell>
                          {version.categories.length > 0
                            ? version.categories.join(", ")
                            : "None"}
                        </TableCell>
                        <TableCell className="text-right">
                          <div className="flex justify-end gap-2">
                            {version.listing_version_id && (
                              <>
                                <Link
                                  href={`/admin/marketplace/preview/${version.listing_version_id}`}
                                >
                                  <Button size="xs" variant="secondary">
                                    <Eye className="h-4 w-4" />
                                    Preview
                                  </Button>
                                </Link>
                                <DownloadAgentAdminButton
                                  storeListingVersionId={
                                    version.listing_version_id
                                  }
                                />
                              </>
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
          </td>
        </TableRow>
      )}
    </>
  );
}
