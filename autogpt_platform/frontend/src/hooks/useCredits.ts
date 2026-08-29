import AutoGPTServerAPI from "@/lib/autogpt-server-api";
import {
  RefundRequest,
  TransactionHistory,
} from "@/lib/autogpt-server-api/types";
import { useAuth } from "@/lib/auth/hooks/useAuth";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";

export default function useCredits({
  identityKey,
  fetchInitialCredits = false,
  fetchInitialAutoTopUpConfig = false,
  fetchInitialTransactionHistory = false,
  fetchInitialRefundRequests = false,
}: {
  identityKey?: string | null;
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
  fetchTransactionHistory: (resetCursor?: boolean) => void;
  refundRequests: RefundRequest[];
  fetchRefundRequests: () => void;
  formatCredits: (credit: number | null) => string;
} {
  const auth = useAuth();
  const resolvedIdentityKey =
    identityKey !== undefined ? identityKey : (auth?.user?.id ?? null);

  const [credits, setCredits] = useState<number | null>(null);
  const [autoTopUpConfig, setAutoTopUpConfig] = useState<{
    amount: number;
    threshold: number;
  } | null>(null);
  const [transactionHistory, setTransactionHistory] =
    useState<TransactionHistory>({
      transactions: [],
      next_transaction_time: null,
    });
  const [refundRequests, setRefundRequests] = useState<RefundRequest[]>([]);

  const identityKeyRef = useRef(resolvedIdentityKey);
  identityKeyRef.current = resolvedIdentityKey;
  const [stateIdentityKey, setStateIdentityKey] = useState(resolvedIdentityKey);
  const cursorRef = useRef<Date | null>(null);

  useEffect(() => {
    identityKeyRef.current = resolvedIdentityKey;
    setStateIdentityKey(resolvedIdentityKey);
    cursorRef.current = null;
    setCredits(null);
    setAutoTopUpConfig(null);
    setTransactionHistory({
      transactions: [],
      next_transaction_time: null,
    });
    setRefundRequests([]);
  }, [resolvedIdentityKey]);

  const api = useMemo(() => new AutoGPTServerAPI(), []);
  const router = useRouter();

  const fetchCredits = useCallback(async () => {
    const requestIdentityKey = resolvedIdentityKey;
    if (!requestIdentityKey) return;
    const response = await api.getUserCredit();
    if (identityKeyRef.current !== requestIdentityKey) return;
    setCredits(response.credits ?? null);
  }, [api, resolvedIdentityKey]);

  useEffect(() => {
    if (!fetchInitialCredits || !resolvedIdentityKey) return;
    fetchCredits();
  }, [fetchCredits, fetchInitialCredits, resolvedIdentityKey]);

  const fetchAutoTopUpConfig = useCallback(async () => {
    const requestIdentityKey = resolvedIdentityKey;
    if (!requestIdentityKey) return;
    const response = await api.getAutoTopUpConfig();
    if (identityKeyRef.current !== requestIdentityKey) return;
    setAutoTopUpConfig(response);
  }, [api, resolvedIdentityKey]);

  useEffect(() => {
    if (!fetchInitialAutoTopUpConfig || !resolvedIdentityKey) return;
    fetchAutoTopUpConfig();
  }, [fetchAutoTopUpConfig, fetchInitialAutoTopUpConfig, resolvedIdentityKey]);

  const updateAutoTopUpConfig = useCallback(
    async (amount: number, threshold: number) => {
      const requestIdentityKey = resolvedIdentityKey;
      if (!requestIdentityKey) return;
      await api.setAutoTopUpConfig({ amount, threshold });
      if (identityKeyRef.current !== requestIdentityKey) return;
      fetchAutoTopUpConfig();
    },
    [api, fetchAutoTopUpConfig, resolvedIdentityKey],
  );

  const requestTopUp = useCallback(
    async (credit_amount: number) => {
      const requestIdentityKey = resolvedIdentityKey;
      if (!requestIdentityKey) return;
      const response = await api.requestTopUp(credit_amount);
      if (identityKeyRef.current !== requestIdentityKey) return;
      router.push(response.checkout_url);
    },
    [api, router, resolvedIdentityKey],
  );

  const refundTopUp = useCallback(
    async (transaction_key: string, reason: string) => {
      const requestIdentityKey = resolvedIdentityKey;
      if (!requestIdentityKey) return 0;
      const refunded_amount = await api.refundTopUp(transaction_key, reason);
      if (identityKeyRef.current !== requestIdentityKey) return refunded_amount;
      await fetchCredits();
      if (identityKeyRef.current !== requestIdentityKey) return refunded_amount;
      const updatedHistory = await api.getTransactionHistory(null, 20);
      if (identityKeyRef.current !== requestIdentityKey) return refunded_amount;
      cursorRef.current = updatedHistory.next_transaction_time;
      setTransactionHistory(updatedHistory);
      return refunded_amount;
    },
    [api, fetchCredits, resolvedIdentityKey],
  );

  const fetchTransactionHistory = useCallback(
    async (resetCursor = false) => {
      const requestIdentityKey = resolvedIdentityKey;
      if (!requestIdentityKey) return;
      const cursor = resetCursor ? null : cursorRef.current;
      const response = await api.getTransactionHistory(cursor, 20);
      if (identityKeyRef.current !== requestIdentityKey) return;
      cursorRef.current = response.next_transaction_time;
      setTransactionHistory((prev) => ({
        transactions: resetCursor
          ? response.transactions
          : [...prev.transactions, ...response.transactions],
        next_transaction_time: response.next_transaction_time,
      }));
    },
    [api, resolvedIdentityKey],
  );

  useEffect(() => {
    if (!fetchInitialTransactionHistory || !resolvedIdentityKey) return;
    fetchTransactionHistory(true);
  }, [
    fetchInitialTransactionHistory,
    resolvedIdentityKey,
    fetchTransactionHistory,
  ]);

  const fetchRefundRequests = useCallback(async () => {
    const requestIdentityKey = resolvedIdentityKey;
    if (!requestIdentityKey) return;
    const response = await api.getRefundRequests();
    if (identityKeyRef.current !== requestIdentityKey) return;
    setRefundRequests(response);
  }, [api, resolvedIdentityKey]);

  useEffect(() => {
    if (!fetchInitialRefundRequests || !resolvedIdentityKey) return;
    fetchRefundRequests();
  }, [fetchInitialRefundRequests, resolvedIdentityKey, fetchRefundRequests]);

  const formatCredits = useCallback((credit: number | null) => {
    if (credit === null) {
      return "-";
    }
    const value = Math.abs(credit);
    const sign = credit < 0 ? "-" : "";
    return `${sign}$${(value / 100).toFixed(2)}`;
  }, []);

  const isCurrentIdentityActive =
    stateIdentityKey === resolvedIdentityKey && resolvedIdentityKey !== null;

  return {
    credits: isCurrentIdentityActive ? credits : null,
    fetchCredits,
    requestTopUp,
    refundTopUp,
    autoTopUpConfig: isCurrentIdentityActive ? autoTopUpConfig : null,
    fetchAutoTopUpConfig,
    updateAutoTopUpConfig,
    transactionHistory: isCurrentIdentityActive
      ? transactionHistory
      : { transactions: [], next_transaction_time: null },
    fetchTransactionHistory,
    refundRequests: isCurrentIdentityActive ? refundRequests : [],
    fetchRefundRequests,
    formatCredits,
  };
}
