"use client";

import { useEffect, useState } from "react";
import { getV2ExportBlockCostEstimates } from "@/app/api/__generated__/endpoints/admin/admin";
import { okData } from "@/app/api/helpers";
import { ApiError } from "@/lib/autogpt-server-api/helpers";
import { useToast } from "@/components/molecules/Toast/use-toast";
import type { BlockCostEstimatesResponse } from "@/app/api/__generated__/models/blockCostEstimatesResponse";
import {
  dateInputToUtcIso,
  dateInputToUtcIsoEnd,
  defaultEndDate,
  defaultStartDate,
} from "../helpers";

export function useBlockCostEstimates() {
  const { toast } = useToast();
  const [start, setStart] = useState(defaultStartDate);
  const [end, setEnd] = useState(defaultEndDate);
  const [minSamples, setMinSamples] = useState(10);
  const [data, setData] = useState<BlockCostEstimatesResponse | null>(null);
  const [loading, setLoading] = useState(false);

  // Re-baseline the default window every mount so a long-lived tab doesn't
  // freeze the dates at the original mount time.
  useEffect(() => {
    setStart(defaultStartDate());
    setEnd(defaultEndDate());
  }, []);

  async function fetchEstimates() {
    if (!start || !end) {
      toast({
        title: "Pick a date range",
        description: "Both start and end dates are required.",
        variant: "destructive",
      });
      return;
    }
    setLoading(true);
    try {
      const startIso = dateInputToUtcIso(start);
      const endIso = dateInputToUtcIsoEnd(end);
      if (!startIso || !endIso) {
        toast({
          title: "Invalid date range",
          description: "Could not parse the selected start/end as UTC.",
          variant: "destructive",
        });
        return;
      }
      const safeMinSamples = Math.max(1, Math.floor(Number(minSamples) || 1));
      const response = await getV2ExportBlockCostEstimates({
        start: startIso as unknown as Date,
        end: endIso as unknown as Date,
        min_samples: safeMinSamples,
      });
      const payload = okData(response);
      if (!payload) {
        toast({
          title: "Aggregation failed",
          description: "Unexpected response shape from the export endpoint.",
          variant: "destructive",
        });
        return;
      }
      setData(payload);
      toast({
        title: "Aggregation complete",
        description: `${payload.total_rows} dynamic-cost blocks aggregated.`,
      });
    } catch (err) {
      if (err instanceof ApiError) {
        const detail =
          (err.response as { detail?: string } | undefined)?.detail ??
          err.message;
        toast({
          title: err.status === 400 ? "Window too large" : "Aggregation failed",
          description: detail,
          variant: "destructive",
        });
      } else {
        toast({
          title: "Aggregation failed",
          description: err instanceof Error ? err.message : "Unknown error",
          variant: "destructive",
        });
      }
    } finally {
      setLoading(false);
    }
  }

  return {
    start,
    end,
    minSamples,
    data,
    loading,
    setStart,
    setEnd,
    setMinSamples,
    fetchEstimates,
  };
}
