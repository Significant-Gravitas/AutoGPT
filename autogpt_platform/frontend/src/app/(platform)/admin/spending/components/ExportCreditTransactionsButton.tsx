"use client";

import { useEffect, useState } from "react";
import { DownloadSimpleIcon } from "@phosphor-icons/react";
import { Button } from "@/components/atoms/Button/Button";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/__legacy__/ui/select";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { getV2ExportCreditTransactions } from "@/app/api/__generated__/endpoints/admin/admin";
import { okData } from "@/app/api/helpers";
import { ApiError } from "@/lib/autogpt-server-api/helpers";
import { CreditTransactionType } from "@/app/api/__generated__/models/creditTransactionType";
import {
  buildCreditTransactionsCsv,
  dateInputToUtcIso,
  dateInputToUtcIsoEnd,
  defaultEndDate,
  defaultStartDate,
  downloadCsv,
} from "../helpers";

type TypeFilter = "ALL" | CreditTransactionType;

const TYPE_OPTIONS: { value: TypeFilter; label: string }[] = [
  { value: "ALL", label: "All types" },
  { value: CreditTransactionType.TOP_UP, label: "Top up" },
  { value: CreditTransactionType.USAGE, label: "Usage" },
  { value: CreditTransactionType.GRANT, label: "Grant" },
  { value: CreditTransactionType.REFUND, label: "Refund" },
  { value: CreditTransactionType.CARD_CHECK, label: "Card check" },
  { value: CreditTransactionType.SUBSCRIPTION, label: "Subscription" },
];

export function ExportCreditTransactionsButton() {
  const { toast } = useToast();
  const [open, setOpen] = useState(false);
  const [start, setStart] = useState(defaultStartDate);
  const [end, setEnd] = useState(defaultEndDate);
  const [typeFilter, setTypeFilter] = useState<TypeFilter>("ALL");
  const [userId, setUserId] = useState("");
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
      const response = await getV2ExportCreditTransactions({
        start: startIso as unknown as Date,
        end: endIso as unknown as Date,
        transaction_type:
          typeFilter === "ALL"
            ? undefined
            : (typeFilter as CreditTransactionType),
        user_id: userId.trim() || undefined,
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
      const csv = buildCreditTransactionsCsv(data.transactions);
      downloadCsv(csv, `credit_transactions_${start}_${end}.csv`);
      toast({
        title: "Export ready",
        description: `${data.total_rows} transactions downloaded.`,
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
      title="Export credit transactions"
      styling={{ maxWidth: "30rem" }}
      controlled={{ isOpen: open, set: setOpen }}
    >
      <Dialog.Trigger>
        <Button
          variant="secondary"
          size="small"
          leftIcon={<DownloadSimpleIcon weight="bold" />}
        >
          Export CSV
        </Button>
      </Dialog.Trigger>
      <Dialog.Content>
        <div className="flex flex-col gap-4">
          <div className="flex gap-3">
            <div className="flex flex-1 flex-col gap-1">
              <label htmlFor="credit-export-start" className="text-sm">
                Start date (UTC)
              </label>
              <input
                id="credit-export-start"
                type="date"
                className="rounded border px-3 py-1.5 text-sm"
                value={start}
                onChange={(e) => setStart(e.target.value)}
              />
            </div>
            <div className="flex flex-1 flex-col gap-1">
              <label htmlFor="credit-export-end" className="text-sm">
                End date (UTC)
              </label>
              <input
                id="credit-export-end"
                type="date"
                className="rounded border px-3 py-1.5 text-sm"
                value={end}
                onChange={(e) => setEnd(e.target.value)}
              />
            </div>
          </div>
          <div className="flex flex-col gap-1">
            <label className="text-sm" htmlFor="credit-export-type">
              Transaction type
            </label>
            <Select
              value={typeFilter}
              onValueChange={(v) => setTypeFilter(v as TypeFilter)}
            >
              <SelectTrigger id="credit-export-type" className="w-full">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {TYPE_OPTIONS.map((opt) => (
                  <SelectItem key={opt.value} value={opt.value}>
                    {opt.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div className="flex flex-col gap-1">
            <label className="text-sm" htmlFor="credit-export-user-id">
              User ID (optional)
            </label>
            <input
              id="credit-export-user-id"
              type="text"
              placeholder="Filter by a single user ID"
              className="rounded border px-3 py-1.5 text-sm"
              value={userId}
              onChange={(e) => setUserId(e.target.value)}
            />
          </div>
          <p className="text-xs text-muted-foreground">
            Window is capped at 90 days and 100k rows. Narrow the range if the
            backend returns a 400.
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
