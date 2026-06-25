"use client";

import { useEffect, useState } from "react";
import { ChartLineIcon } from "@phosphor-icons/react";
import { Button } from "@/components/atoms/Button/Button";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { getV2ExportCopilotWeeklyUsageVsRateLimit } from "@/app/api/__generated__/endpoints/admin/admin";
import { okData } from "@/app/api/helpers";
import { ApiError } from "@/lib/autogpt-server-api/helpers";
import {
  buildCopilotUsageCsv,
  dateInputToUtcIso,
  dateInputToUtcIsoEnd,
  defaultEndDate,
  defaultStartDate,
  downloadCsv,
} from "../helpers";

export function ExportCopilotUsageButton() {
  const { toast } = useToast();
  const [open, setOpen] = useState(false);
  const [start, setStart] = useState(defaultStartDate);
  const [end, setEnd] = useState(defaultEndDate);
  const [exporting, setExporting] = useState(false);

  // Refresh the default window each time the dialog opens — the lazy useState
  // initializer only runs once on mount, so without this the dates would
  // freeze at component-mount time and drift on a long-lived page.
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
      const response = await getV2ExportCopilotWeeklyUsageVsRateLimit({
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
      const csv = buildCopilotUsageCsv(data.rows);
      downloadCsv(csv, `copilot_weekly_usage_${start}_${end}.csv`);
      toast({
        title: "Export ready",
        description: `${data.total_rows} (user, week) rows downloaded.`,
      });
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
      title="Export copilot weekly usage"
      styling={{ maxWidth: "30rem" }}
      controlled={{ isOpen: open, set: setOpen }}
    >
      <Dialog.Trigger>
        <Button
          variant="secondary"
          size="small"
          leftIcon={<ChartLineIcon weight="bold" />}
        >
          Copilot Usage CSV
        </Button>
      </Dialog.Trigger>
      <Dialog.Content>
        <div className="flex flex-col gap-4">
          <p className="text-sm text-muted-foreground">
            Aggregates copilot:* spend by user and ISO week and joins each row
            against the user&apos;s tier-derived weekly limit.
          </p>
          <div className="flex gap-3">
            <div className="flex flex-1 flex-col gap-1">
              <label htmlFor="copilot-export-start" className="text-sm">
                Start date (UTC)
              </label>
              <input
                id="copilot-export-start"
                type="date"
                className="rounded border px-3 py-1.5 text-sm"
                value={start}
                onChange={(e) => setStart(e.target.value)}
              />
            </div>
            <div className="flex flex-1 flex-col gap-1">
              <label htmlFor="copilot-export-end" className="text-sm">
                End date (UTC)
              </label>
              <input
                id="copilot-export-end"
                type="date"
                className="rounded border px-3 py-1.5 text-sm"
                value={end}
                onChange={(e) => setEnd(e.target.value)}
              />
            </div>
          </div>
          <p className="text-xs text-muted-foreground">
            Window is capped at 90 days and 100k rows.
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
