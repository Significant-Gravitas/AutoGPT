import AutoGPTServerAPI from "@/lib/autogpt-server-api";
import { useCallback, useEffect, useMemo, useState } from "react";

export default function useCredits(): {
  credits: number | null;
  fetchCredits: () => void;
} {
  const [credits, setCredits] = useState<number | null>(null);
  const api = useMemo(() => new AutoGPTServerAPI(), []);

  const fetchCredits = useCallback(async () => {
    const response = await api.getUserCredit();
    setCredits(response.credits);
  }, []);

  useEffect(() => {
    fetchCredits();
  }, [fetchCredits]);

  return {
    credits,
    fetchCredits,
  };
}
