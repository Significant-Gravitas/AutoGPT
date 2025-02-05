import AutoGPTServerAPI from "@/lib/autogpt-server-api";
import { TransactionHistory } from "@/lib/autogpt-server-api/types";
import { useCallback, useEffect, useMemo, useState } from "react";
import { loadStripe } from "@stripe/stripe-js";
import { useRouter } from "next/navigation";

const stripePromise = loadStripe(
  process.env.NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY!,
);

export default function useCredits(): {
  credits: number | null;
  fetchCredits: () => void;
  requestTopUp: (credit_amount: number) => Promise<void>;
  autoTopUpConfig: { amount: number; threshold: number } | null;
  fetchAutoTopUpConfig: () => void;
  updateAutoTopUpConfig: (amount: number, threshold: number) => Promise<void>;
  transactionHistory: TransactionHistory;
  fetchTransactionHistory: () => void;
  formatCredits: (credit: number | null) => string;
} {
  const [credits, setCredits] = useState<number | null>(null);
  const [autoTopUpConfig, setAutoTopUpConfig] = useState<{
    amount: number;
    threshold: number;
  } | null>(null);

  const api = useMemo(() => new AutoGPTServerAPI(), []);
  const router = useRouter();

  const fetchCredits = useCallback(async () => {
    const response = await api.getUserCredit();
    setCredits(response.credits);
  }, [api]);

  useEffect(() => {
    fetchCredits();
  }, [fetchCredits]);

  const fetchAutoTopUpConfig = useCallback(async () => {
    const response = await api.getAutoTopUpConfig();
    setAutoTopUpConfig(response);
  }, [api]);

  useEffect(() => {
    fetchAutoTopUpConfig();
  }, [fetchAutoTopUpConfig]);

  const updateAutoTopUpConfig = useCallback(
    async (amount: number, threshold: number) => {
      await api.setAutoTopUpConfig({ amount, threshold });
      fetchAutoTopUpConfig();
    },
    [api, fetchAutoTopUpConfig],
  );

  const requestTopUp = useCallback(
    async (credit_amount: number) => {
      const stripe = await stripePromise;

      if (!stripe) {
        return;
      }

      const response = await api.requestTopUp(credit_amount);
      router.push(response.checkout_url);
    },
    [api, router],
  );

  const [transactionHistory, setTransactionHistory] =
    useState<TransactionHistory>({
      transactions: [],
      next_transaction_time: null,
    });

  const fetchTransactionHistory = useCallback(async () => {
    const response = await api.getTransactionHistory(
      transactionHistory.next_transaction_time,
      20,
    );
    setTransactionHistory({
      transactions: [
        ...transactionHistory.transactions,
        ...response.transactions,
      ],
      next_transaction_time: response.next_transaction_time,
    });
  }, [api, transactionHistory]);

  useEffect(() => {
    fetchTransactionHistory();
    // Note: We only need to fetch transaction history once.
    // Hence, we should avoid `fetchTransactionHistory` to the dependency array.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const formatCredits = useCallback((credit: number | null) => {
    if (credit === null) {
      return "-";
    }
    const value = Math.abs(credit);
    const sign = credit < 0 ? "-" : "";
    const precision =
      2 - (value % 100 === 0 ? 1 : 0) - (value % 10 === 0 ? 1 : 0);
    return `${sign}$${(value / 100).toFixed(precision)}`;
  }, []);

  return {
    credits,
    fetchCredits,
    requestTopUp,
    autoTopUpConfig,
    fetchAutoTopUpConfig,
    updateAutoTopUpConfig,
    transactionHistory,
    fetchTransactionHistory,
    formatCredits,
  };
}
