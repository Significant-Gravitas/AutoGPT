"use client";

import { getV2ExportCopilotWeeklyUsageVsRateLimit } from "@/app/api/__generated__/endpoints/admin/admin";
import { buildCopilotUsageCsv } from "../helpers";
import { ChartLineIcon } from "@hugeicons/core-free-icons";
import { ExportCSVDialog } from "./ExportCSVDialog";

export function ExportCopilotUsageButton() {
  return (
    <ExportCSVDialog
      title="Export copilot weekly usage"
      triggerLabel="Copilot Usage CSV"
      triggerIcon={ChartLineIcon}
      idPrefix="copilot-export"
      filePrefix="copilot_weekly_usage"
      description="Aggregates copilot:* spend by user and ISO week and joins each row against the user's tier-derived weekly limit."
      fetchExport={getV2ExportCopilotWeeklyUsageVsRateLimit}
      toCSV={(data) => buildCopilotUsageCsv(data.rows)}
      toSummary={(data) => `${data.total_rows} (user, week) rows downloaded.`}
    />
  );
}
