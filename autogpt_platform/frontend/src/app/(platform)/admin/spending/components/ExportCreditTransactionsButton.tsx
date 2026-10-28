"use client";

import { useState } from "react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/atoms/Select/components/SelectPrimitives";
import { getV2ExportCreditTransactions } from "@/app/api/__generated__/endpoints/admin/admin";
import { CreditTransactionType } from "@/app/api/__generated__/models/creditTransactionType";
import { buildCreditTransactionsCsv } from "../helpers";
import { Download04Icon } from "@hugeicons/core-free-icons";
import { ExportCSVDialog } from "./ExportCSVDialog";

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
  const [typeFilter, setTypeFilter] = useState<TypeFilter>("ALL");
  const [userId, setUserId] = useState("");

  return (
    <ExportCSVDialog
      title="Export credit transactions"
      triggerLabel="Export CSV"
      triggerIcon={Download04Icon}
      idPrefix="credit-export"
      filePrefix="credit_transactions"
      capNoteSuffix="Narrow the range if the backend returns a 400."
      fetchExport={(range) =>
        getV2ExportCreditTransactions({
          ...range,
          transaction_type:
            typeFilter === "ALL"
              ? undefined
              : (typeFilter as CreditTransactionType),
          user_id: userId.trim() || undefined,
        })
      }
      toCSV={(data) => buildCreditTransactionsCsv(data.transactions)}
      toSummary={(data) => `${data.total_rows} transactions downloaded.`}
    >
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
    </ExportCSVDialog>
  );
}
