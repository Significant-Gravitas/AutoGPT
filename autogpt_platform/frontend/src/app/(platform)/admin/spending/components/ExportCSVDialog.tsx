import { useEffect, useState, type ReactNode } from "react";
import type { IconSvgElement } from "@hugeicons/react";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { okData, type OKData } from "@/app/api/helpers";
import { ApiError } from "@/lib/autogpt-server-api/helpers";
import {
  dateInputToUtcIso,
  dateInputToUtcIsoEnd,
  defaultEndDate,
  defaultStartDate,
  downloadCsv,
} from "../helpers";

const CAP_NOTE = "Window is capped at 90 days and 100k rows.";

interface Props<TResponse extends { status: number; data?: unknown }> {
  title: string;
  triggerLabel: string;
  triggerIcon: IconSvgElement;
  idPrefix: string;
  filePrefix: string;
  description?: string;
  capNoteSuffix?: string;
  fetchExport: (range: { start: Date; end: Date }) => Promise<TResponse>;
  toCSV: (data: OKData<TResponse>) => string;
  toSummary: (data: OKData<TResponse>) => string;
  children?: ReactNode;
}

export function ExportCSVDialog<
  TResponse extends { status: number; data?: unknown },
>({
  title,
  triggerLabel,
  triggerIcon,
  idPrefix,
  filePrefix,
  description,
  capNoteSuffix,
  fetchExport,
  toCSV,
  toSummary,
  children,
}: Props<TResponse>) {
  const { toast } = useToast();
  const [open, setOpen] = useState(false);
  const [start, setStart] = useState(defaultStartDate);
  const [end, setEnd] = useState(defaultEndDate);
  const [exporting, setExporting] = useState(false);

  // The lazy initializers run once on mount; refresh the default window on
  // every open so a long-lived page does not export a stale range.
  useEffect(() => {
    if (open) {
      setStart(defaultStartDate());
      setEnd(defaultEndDate());
    }
  }, [open]);

  async function handleExport() {
    if (!start || !end) {
      toast({
        title: "Pick a date range",
        description: "Both start and end dates are required.",
        variant: "destructive",
      });
      return;
    }
    setExporting(true);
    try {
      const startIso = dateInputToUtcIso(start);
      const endIso = dateInputToUtcIsoEnd(end);
      if (!startIso || !endIso) return;
      // Pass ISO strings as-is — orval calls .toString() on Date params, which
      // produces a localised string FastAPI rejects (422). Strings round-trip.
      const response = await fetchExport({
        start: startIso as unknown as Date,
        end: endIso as unknown as Date,
      });
      const data = okData(response);
      if (!data) {
        toast({
          title: "Export failed",
          description: "Unexpected response shape from the export endpoint.",
          variant: "destructive",
        });
        return;
      }
      downloadCsv(toCSV(data), `${filePrefix}_${start}_${end}.csv`);
      toast({ title: "Export ready", description: toSummary(data) });
      setOpen(false);
    } catch (err) {
      // customMutator throws ApiError on non-2xx. Surface backend cap-exceed
      // detail (400) so the operator can narrow their range without a refresh.
      if (err instanceof ApiError) {
        const detail =
          (err.response as { detail?: string } | undefined)?.detail ??
          err.message;
        toast({
          title: err.status === 400 ? "Window too large" : "Export failed",
          description: detail,
          variant: "destructive",
        });
      } else {
        toast({
          title: "Export failed",
          description: err instanceof Error ? err.message : "Unknown error",
          variant: "destructive",
        });
      }
    } finally {
      setExporting(false);
    }
  }

  return (
    <Dialog
      title={title}
      styling={{ maxWidth: "30rem" }}
      controlled={{ isOpen: open, set: setOpen }}
    >
      <Dialog.Trigger>
        <Button
          variant="secondary"
          size="small"
          leftIcon={<Icon icon={triggerIcon} />}
        >
          {triggerLabel}
        </Button>
      </Dialog.Trigger>
      <Dialog.Content>
        <div className="flex flex-col gap-4">
          {description && (
            <p className="text-sm text-muted-foreground">{description}</p>
          )}
          <div className="flex gap-3">
            <div className="flex flex-1 flex-col gap-1">
              <label htmlFor={`${idPrefix}-start`} className="text-sm">
                Start date (UTC)
              </label>
              <input
                id={`${idPrefix}-start`}
                type="date"
                className="rounded border px-3 py-1.5 text-sm"
                value={start}
                onChange={(e) => setStart(e.target.value)}
              />
            </div>
            <div className="flex flex-1 flex-col gap-1">
              <label htmlFor={`${idPrefix}-end`} className="text-sm">
                End date (UTC)
              </label>
              <input
                id={`${idPrefix}-end`}
                type="date"
                className="rounded border px-3 py-1.5 text-sm"
                value={end}
                onChange={(e) => setEnd(e.target.value)}
              />
            </div>
          </div>
          {children}
          <p className="text-xs text-muted-foreground">
            {capNoteSuffix ? `${CAP_NOTE} ${capNoteSuffix}` : CAP_NOTE}
          </p>
        </div>
        <Dialog.Footer>
          <Button
            variant="secondary"
            size="small"
            onClick={() => setOpen(false)}
            disabled={exporting}
          >
            Cancel
          </Button>
          <Button
            variant="primary"
            size="small"
            onClick={handleExport}
            loading={exporting}
          >
            Download CSV
          </Button>
        </Dialog.Footer>
      </Dialog.Content>
    </Dialog>
  );
}
