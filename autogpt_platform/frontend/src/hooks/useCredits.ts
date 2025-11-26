import AutoGPTServerAPI from "@/lib/autogpt-server-api";
import {
  RefundRequest,
  TransactionHistory,
} from "@/lib/autogpt-server-api/types";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useMemo, useState } from "react";

export default function useCredits({
  fetchInitialCredits = false,
  fetchInitialAutoTopUpConfig = false,
  fetchInitialTransactionHistory = false,
  fetchInitialRefundRequests = false,
}: {
  fetchInitialCredits?: boolean;
  fetchInitialAutoTopUpConfig?: boolean;
  fetchInitialTransactionHistory?: boolean;
  fetchInitialRefundRequests?: boolean;
} = {}): {
  credits: number | null;
  fetchCredits: () => void;
  requestTopUp: (credit_amount: number) => Promise<void>;
  refundTopUp: (transaction_key: string, reason: string) => Promise<number>;
  autoTopUpConfig: { amount: number; threshold: number } | null;
  fetchAutoTopUpConfig: () => void;
  updateAutoTopUpConfig: (amount: number, threshold: number) => Promise<void>;
  transactionHistory: TransactionHistory;
  fetchTransactionHistory: () => void;
  refundRequests: RefundRequest[];
  fetchRefundRequests: () => void;
  formatCredits: (credit: number | null) => string;
} {
  const { isLoggedIn } = useSupabase();
  const [credits, setCredits] = useState<number | null>(null);
  const [autoTopUpConfig, setAutoTopUpConfig] = useState<{
    amount: number;
    threshold: number;
  } | null>(null);

  const api = useMemo(() => new AutoGPTServerAPI(), []);
  const router = useRouter();

  const fetchCredits = useCallback(async () => {
    if (!isLoggedIn) return;
    try {
      const response = await api.getUserCredit();
      if (response) {
        setCredits(response.credits);
      } else {
        setCredits(null);
      }
    } catch (error) {
      console.error("Error fetching credits:", error);
      setCredits(null);
    }
  }, [api, isLoggedIn]);

  useEffect(() => {
    if (!fetchInitialCredits || !isLoggedIn) return;
    fetchCredits();
  }, [fetchCredits, fetchInitialCredits, isLoggedIn]);

  // Clear credits when user logs out
  useEffect(() => {
    if (!isLoggedIn) {
      setCredits(null);
    }
  }, [isLoggedIn]);

  const fetchAutoTopUpConfig = useCallback(async () => {
    if (!isLoggedIn) return;
    try {
      const response = await api.getAutoTopUpConfig();
      setAutoTopUpConfig(response || null);
    } catch (error) {
      console.error("Error fetching auto top-up config:", error);
      setAutoTopUpConfig(null);
    }
  }, [api, isLoggedIn]);

  useEffect(() => {
    if (!fetchInitialAutoTopUpConfig || !isLoggedIn) return;
    fetchAutoTopUpConfig();
  }, [fetchAutoTopUpConfig, fetchInitialAutoTopUpConfig, isLoggedIn]);

  // Clear auto top-up config when user logs out
  useEffect(() => {
    if (!isLoggedIn) {
      setAutoTopUpConfig(null);
    }
  }, [isLoggedIn]);

  const updateAutoTopUpConfig = useCallback(
    async (amount: number, threshold: number) => {
      if (!isLoggedIn) return;
      await api.setAutoTopUpConfig({ amount, threshold });
      fetchAutoTopUpConfig();
    },
    [api, fetchAutoTopUpConfig, isLoggedIn],
  );

  const requestTopUp = useCallback(
    async (credit_amount: number) => {
      if (!isLoggedIn) return;
      const response = await api.requestTopUp(credit_amount);
      router.push(response.checkout_url);
    },
    [api, router, isLoggedIn],
  );

  const refundTopUp = useCallback(
    async (transaction_key: string, reason: string) => {
      if (!isLoggedIn) return 0;
      try {
        const refunded_amount = await api.refundTopUp(transaction_key, reason);
        await fetchCredits();
        const history = await api.getTransactionHistory();
        if (history) {
          setTransactionHistory(history);
        }
        return refunded_amount;
      } catch (error) {
        console.error("Error refunding top-up:", error);
        throw error;
      }
    },
    [api, fetchCredits, isLoggedIn],
  );

  const [transactionHistory, setTransactionHistory] =
    useState<TransactionHistory>({
      transactions: [],
      next_transaction_time: null,
    });

  const fetchTransactionHistory = useCallback(async () => {
    if (!isLoggedIn) return;
    try {
      const response = await api.getTransactionHistory(
        transactionHistory.next_transaction_time,
        20,
      );
      if (response) {
        setTransactionHistory({
          transactions: [
            ...transactionHistory.transactions,
            ...response.transactions,
          ],
          next_transaction_time: response.next_transaction_time,
        });
      }
    } catch (error) {
      console.error("Error fetching transaction history:", error);
    }
  }, [api, transactionHistory, isLoggedIn]);

  useEffect(() => {
    if (!fetchInitialTransactionHistory || !isLoggedIn) return;
    fetchTransactionHistory();
    // Note: We only need to fetch transaction history once.
    // Hence, we should avoid `fetchTransactionHistory` to the dependency array.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fetchInitialTransactionHistory, isLoggedIn]);

  // Clear transaction history when user logs out
  useEffect(() => {
    if (!isLoggedIn) {
      setTransactionHistory({
        transactions: [],
        next_transaction_time: null,
      });
    }
  }, [isLoggedIn]);

  const [refundRequests, setRefundRequests] = useState<RefundRequest[]>([]);

  const fetchRefundRequests = useCallback(async () => {
    if (!isLoggedIn) return;
    try {
      const response = await api.getRefundRequests();
      setRefundRequests(response || []);
    } catch (error) {
      console.error("Error fetching refund requests:", error);
      setRefundRequests([]);
    }
  }, [api, isLoggedIn]);

  useEffect(() => {
    if (!fetchInitialRefundRequests || !isLoggedIn) return;
    fetchRefundRequests();
  }, [fetchRefundRequests, fetchInitialRefundRequests, isLoggedIn]);

  // Clear refund requests when user logs out
  useEffect(() => {
    if (!isLoggedIn) {
      setRefundRequests([]);
    }
  }, [isLoggedIn]);

  const formatCredits = useCallback((credit: number | null) => {
    if (credit === null) {
      return "-";
    }
    const value = Math.abs(credit);
    const sign = credit < 0 ? "-" : "";
    return `${sign}$${(value / 100).toFixed(2)}`;
  }, []);

  return {
    credits,
    fetchCredits,
    requestTopUp,
    refundTopUp,
    autoTopUpConfig,
    fetchAutoTopUpConfig,
    updateAutoTopUpConfig,
    transactionHistory,
    fetchTransactionHistory,
    refundRequests,
    fetchRefundRequests,
    formatCredits,
  };
}
