"use client";

import { useGetV1GetExecutionDetails } from "@/app/api/__generated__/endpoints/graphs/graphs";
import { okData } from "@/app/api/helpers";
import { MessageResponse } from "@/components/ai-elements/message";
import { Button } from "@/components/atoms/Button/Button";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import Link from "next/link";
import {
  asTableRows,
  cellText,
  downloadCsv,
  MAX_PREVIEW_COLUMNS,
  MAX_PREVIEW_ROWS,
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
      <div className="space-y-3" data-testid="work-output-loading">
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

  if (outputType === "image" && typeof primary === "string") {
    return (
      // Run outputs are arbitrary external URLs — next/image can't optimize them.
      // eslint-disable-next-line @next/next/no-img-element
      <img
        src={primary}
        alt={title}
        className="max-h-[70vh] w-full rounded-xl object-contain"
      />
    );
  }

  if (outputType === "doc" && typeof primary === "string") {
    return (
      <div className="prose prose-sm max-w-none">
        <MessageResponse>{primary}</MessageResponse>
      </div>
    );
  }

  return <RunLinkFallback runLink={runLink} />;
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
  const allColumns = tableColumns(rows);
  const columns = allColumns.slice(0, MAX_PREVIEW_COLUMNS);
  const truncated =
    rows.length > visibleRows.length || allColumns.length > columns.length;
  return (
    <div className="space-y-3">
      <div className="flex justify-end">
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
      <div className="overflow-x-auto rounded-xl ring-1 ring-inset ring-zinc-200">
        <table className="w-full border-collapse text-sm">
          <thead>
            <tr className="bg-zinc-50 text-left">
              {columns.map((column) => (
                <th
                  key={column}
                  className="px-3 py-2 font-medium text-zinc-600"
                >
                  {column}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {visibleRows.map((row, index) => (
              <tr key={index} className="border-t border-zinc-100">
                {columns.map((column) => (
                  <td key={column} className="px-3 py-2 text-zinc-700">
                    {cellText(row[column])}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {truncated ? (
        <Text variant="small" className="text-zinc-500">
          Showing the first {visibleRows.length} of {rows.length} rows and{" "}
          {columns.length} of {allColumns.length} columns. The CSV export
          matches this preview
          {runLink ? (
            <>
              {" — "}
              <Link href={runLink} className="underline">
                open the full run
              </Link>{" "}
              for everything
            </>
          ) : null}
          .
        </Text>
      ) : null}
    </div>
  );
}

function RunLinkFallback({ runLink }: { runLink?: string | null }) {
  if (!runLink) {
    return (
      <Text variant="body" className="text-zinc-500">
        This run has no preview available.
      </Text>
    );
  }
  return (
    <div className="space-y-3">
      <Text variant="body" className="text-zinc-500">
        Open the full run to inspect its output.
      </Text>
      <Button as="NextLink" href={runLink} variant="primary" size="small">
        Open run details
      </Button>
    </div>
  );
}
