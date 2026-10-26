"use client";

import { useGetV1GetExecutionDetails } from "@/app/api/__generated__/endpoints/graphs/graphs";
import { okData } from "@/app/api/helpers";
import { MessageResponse } from "@/components/organisms/Message/Message";
import { Button } from "@/components/atoms/Button/Button";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/molecules/Sheet/Sheet";
import Link from "next/link";
import { useState } from "react";
import {
  asTableRows,
  cellText,
  downloadCsv,
  MAX_PREVIEW_COLUMNS,
  MAX_PREVIEW_ROWS,
  isSafeImageUrl,
  pickOutputForType,
  tableColumns,
  toCsv,
  type OutputType,
} from "./helpers";

interface Props {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  title: string;
  outputType: OutputType;
  outputKey?: string | null;
  graphId: string;
  executionId: string;
  runLink?: string | null;
}

export function WorkOutputSheet({
  open,
  onOpenChange,
  title,
  outputType,
  outputKey,
  graphId,
  executionId,
  runLink,
}: Props) {
  const shouldFetch = open && outputType !== "unknown";
  const detailsQuery = useGetV1GetExecutionDetails(graphId, executionId, {
    query: {
      select: (res) => okData(res) ?? null,
      enabled: shouldFetch,
    },
  });

  const outputs = detailsQuery.data?.outputs ?? null;
  const primary = outputs
    ? pickOutputForType(
        outputs as Record<string, unknown[]>,
        outputType,
        outputKey,
      )
    : null;

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent
        side="right"
        className="flex w-full flex-col gap-4 overflow-y-auto sm:max-w-xl"
      >
        <SheetHeader className="text-left">
          <SheetTitle className="truncate">{title}</SheetTitle>
          <SheetDescription className="sr-only">
            This run&apos;s output.
          </SheetDescription>
        </SheetHeader>
        <WorkOutputBody
          outputType={outputType}
          title={title}
          primary={primary}
          isLoading={shouldFetch && detailsQuery.isLoading}
          isError={detailsQuery.isError}
          runLink={runLink}
        />
      </SheetContent>
    </Sheet>
  );
}

interface BodyProps {
  outputType: OutputType;
  title: string;
  primary: unknown;
  isLoading: boolean;
  isError: boolean;
  runLink?: string | null;
}

function WorkOutputBody({
  outputType,
  title,
  primary,
  isLoading,
  isError,
  runLink,
}: BodyProps) {
  if (outputType === "unknown") {
    return <RunLinkFallback runLink={runLink} />;
  }

  if (isLoading) {
    return (
      <div
        data-slot="work-output-sheet-loading"
        className="space-y-3"
        data-testid="work-output-loading"
      >
        <Skeleton className="h-6 w-2/3" />
        <Skeleton className="h-40 w-full" />
      </div>
    );
  }

  if (isError || primary == null) {
    return <RunLinkFallback runLink={runLink} />;
  }

  if (outputType === "table") {
    const rows = asTableRows(primary);
    if (!rows) return <RunLinkFallback runLink={runLink} />;
    return <OutputTable title={title} rows={rows} runLink={runLink} />;
  }

  if (outputType === "image") {
    return <OutputImage source={primary} title={title} runLink={runLink} />;
  }

  if (outputType === "doc" && typeof primary === "string") {
    return (
      <div
        data-slot="work-output-sheet-doc"
        className="prose prose-sm max-w-none"
      >
        <MessageResponse>{primary}</MessageResponse>
      </div>
    );
  }

  return <RunLinkFallback runLink={runLink} />;
}

function OutputImage({
  source,
  title,
  runLink,
}: {
  source: unknown;
  title: string;
  runLink?: string | null;
}) {
  const [failed, setFailed] = useState(false);
  if (!isSafeImageUrl(source) || failed) {
    return <RunLinkFallback runLink={runLink} />;
  }
  return (
    // Run outputs are arbitrary external URLs — next/image can't optimize them.
    // eslint-disable-next-line @next/next/no-img-element
    <img
      src={source}
      alt={title}
      referrerPolicy="no-referrer"
      onError={() => setFailed(true)}
      data-slot="work-output-sheet-image"
      className="max-h-[70vh] w-full rounded-xl object-contain"
    />
  );
}

function OutputTable({
  title,
  rows,
  runLink,
}: {
  title: string;
  rows: Record<string, unknown>[];
  runLink?: string | null;
}) {
  const visibleRows = rows.slice(0, MAX_PREVIEW_ROWS);
  const allColumns = tableColumns(visibleRows);
  const columns = allColumns.slice(0, MAX_PREVIEW_COLUMNS);
  const truncated =
    rows.length > visibleRows.length || allColumns.length > columns.length;

  return (
    <div data-slot="work-output-sheet-table" className="space-y-3">
      <div
        data-slot="work-output-sheet-table-actions"
        className="flex justify-end"
      >
        <Button
          variant="secondary"
          size="small"
          onClick={() =>
            downloadCsv(`${title || "run"}.csv`, toCsv(visibleRows, columns))
          }
        >
          Export CSV
        </Button>
      </div>
      <OutputTablePreview rows={visibleRows} columns={columns} />
      {truncated ? (
        <OutputTableTruncationNotice
          visibleRowCount={visibleRows.length}
          rowCount={rows.length}
          visibleColumnCount={columns.length}
          columnCount={allColumns.length}
          runLink={runLink}
        />
      ) : null}
    </div>
  );
}

interface OutputTablePreviewProps {
  rows: Record<string, unknown>[];
  columns: string[];
}

function OutputTablePreview({ rows, columns }: OutputTablePreviewProps) {
  return (
    <div
      data-slot="work-output-sheet-table-preview"
      className="overflow-x-auto rounded-xl ring-1 ring-inset ring-border-base"
    >
      <table
        data-slot="work-output-sheet-table-grid"
        className="w-full border-collapse"
      >
        <thead data-slot="work-output-sheet-table-head">
          <tr
            data-slot="work-output-sheet-table-head-row"
            className="bg-surface-muted text-left"
          >
            {columns.map((column) => (
              <th
                key={column}
                data-slot="work-output-sheet-table-header-cell"
                className="px-3 py-2 text-foreground-secondary"
              >
                <Text as="div" variant="body-medium-tight" tone="secondary">
                  {column}
                </Text>
              </th>
            ))}
          </tr>
        </thead>
        <tbody data-slot="work-output-sheet-table-body">
          {rows.map((row, index) => (
            <tr
              key={index}
              data-slot="work-output-sheet-table-row"
              className="border-t border-separator"
            >
              {columns.map((column) => (
                <td
                  key={column}
                  data-slot="work-output-sheet-table-cell"
                  className="px-3 py-2"
                >
                  <Text as="div" variant="body-tight" tone="body">
                    {cellText(row[column])}
                  </Text>
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

interface OutputTableTruncationNoticeProps {
  visibleRowCount: number;
  rowCount: number;
  visibleColumnCount: number;
  columnCount: number;
  runLink?: string | null;
}

function OutputTableTruncationNotice({
  visibleRowCount,
  rowCount,
  visibleColumnCount,
  columnCount,
  runLink,
}: OutputTableTruncationNoticeProps) {
  return (
    <Text variant="small" tone="muted">
      Showing the first {visibleRowCount} of {rowCount} rows and{" "}
      {visibleColumnCount} of {columnCount} columns. The CSV export matches this
      preview
      {runLink ? (
        <>
          {" — "}
          <Link
            href={runLink}
            data-slot="work-output-sheet-truncation-link"
            className="underline"
          >
            open the full run
          </Link>{" "}
          for everything
        </>
      ) : null}
      .
    </Text>
  );
}

function RunLinkFallback({ runLink }: { runLink?: string | null }) {
  if (!runLink) {
    return (
      <Text variant="body" tone="muted">
        This run has no preview available.
      </Text>
    );
  }
  return (
    <div data-slot="work-output-sheet-fallback" className="space-y-3">
      <Text variant="body" tone="muted">
        Open the full run to inspect its output.
      </Text>
      <Button as="NextLink" href={runLink} variant="primary" size="small">
        Open run details
      </Button>
    </div>
  );
}
