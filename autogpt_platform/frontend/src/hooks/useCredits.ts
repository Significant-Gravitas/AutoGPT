import AutoGPTServerAPI from "@/lib/autogpt-server-api";
import {
  RefundRequest,
  TransactionHistory,
} from "@/lib/autogpt-server-api/types";
import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { useRouter } from "next/navigation";

const EMPTY_TRANSACTION_HISTORY: TransactionHistory = {
  transactions: [],
  next_transaction_time: null,
};
const EMPTY_REFUND_REQUESTS: RefundRequest[] = [];

interface BillingState {
  identityScope: symbol;
  credits: number | null;
  autoTopUpConfig: { amount: number; threshold: number } | null;
  transactionHistory: TransactionHistory;
  refundRequests: RefundRequest[];
}

function emptyBillingState(identityScope: symbol): BillingState {
  return {
    identityScope,
    credits: null,
    autoTopUpConfig: null,
    transactionHistory: EMPTY_TRANSACTION_HISTORY,
    refundRequests: EMPTY_REFUND_REQUESTS,
  };
}

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
  openBillingPortal: () => Promise<void>;
  refundTopUp: (
    transaction_key: string,
    reason: string,
  ) => Promise<number | null>;
  autoTopUpConfig: { amount: number; threshold: number } | null;
  fetchAutoTopUpConfig: () => void;
  updateAutoTopUpConfig: (
    amount: number,
    threshold: number,
  ) => Promise<boolean>;
  transactionHistory: TransactionHistory;
  fetchTransactionHistory: () => void;
  refundRequests: RefundRequest[];
  fetchRefundRequests: () => void;
  formatCredits: (credit: number | null) => string;
} {
  const scopedIdentityKey = identityKey ?? null;
  const identityScopeRef = useRef({
    identityKey: scopedIdentityKey,
    token: Symbol(),
  });
  if (identityScopeRef.current.identityKey !== scopedIdentityKey) {
    identityScopeRef.current = {
      identityKey: scopedIdentityKey,
      token: Symbol(),
    };
  }
  const identityScope = identityScopeRef.current.token;
  const [billingState, setBillingState] = useState(() =>
    emptyBillingState(identityScope),
  );

  useLayoutEffect(() => {
    identityScopeRef.current = {
      identityKey: scopedIdentityKey,
      token: identityScope,
    };
    return () => {
      if (identityScopeRef.current.token !== identityScope) return;
      identityScopeRef.current = { identityKey: null, token: Symbol() };
    };
  }, [identityScope, scopedIdentityKey]);

  useEffect(() => {
    setBillingState((current) =>
      current.identityScope === identityScope
        ? current
        : emptyBillingState(identityScope),
    );
  }, [identityScope]);

  const isCurrentIdentity = billingState.identityScope === identityScope;
  const credits = isCurrentIdentity ? billingState.credits : null;
  const autoTopUpConfig = isCurrentIdentity
    ? billingState.autoTopUpConfig
    : null;
  const transactionHistory = isCurrentIdentity
    ? billingState.transactionHistory
    : EMPTY_TRANSACTION_HISTORY;
  const refundRequests = isCurrentIdentity
    ? billingState.refundRequests
    : EMPTY_REFUND_REQUESTS;

  const api = useMemo(() => new AutoGPTServerAPI(), []);
  const router = useRouter();

  const fetchCredits = useCallback(async () => {
    const requestScope = identityScope;
    if (
      scopedIdentityKey === null ||
      identityScopeRef.current.token !== requestScope
    ) {
      return;
    }
    const response = await api.getUserCredit();
    setBillingState((current) =>
      current.identityScope === requestScope
        ? { ...current, credits: response.credits ?? null }
        : current,
    );
  }, [api, scopedIdentityKey, identityScope]);

  useEffect(() => {
    if (!fetchInitialCredits) return;
    fetchCredits();
  }, [fetchCredits, fetchInitialCredits]);

  const fetchAutoTopUpConfig = useCallback(async () => {
    const requestScope = identityScope;
    if (
      scopedIdentityKey === null ||
      identityScopeRef.current.token !== requestScope
    ) {
      return;
    }
    const response = await api.getAutoTopUpConfig();
    setBillingState((current) =>
      current.identityScope === requestScope
        ? { ...current, autoTopUpConfig: response }
        : current,
    );
  }, [api, scopedIdentityKey, identityScope]);

  useEffect(() => {
    if (!fetchInitialAutoTopUpConfig) return;
    fetchAutoTopUpConfig();
  }, [fetchAutoTopUpConfig, fetchInitialAutoTopUpConfig]);

  const updateAutoTopUpConfig = useCallback(
    async (amount: number, threshold: number) => {
      if (scopedIdentityKey === null) {
        throw new Error("Authentication required");
      }
      const requestScope = identityScope;
      if (identityScopeRef.current.token !== requestScope) return false;
      try {
        await api.setAutoTopUpConfig({ amount, threshold });
      } catch (error) {
        if (identityScopeRef.current.token !== requestScope) return false;
        throw error;
      }
      if (identityScopeRef.current.token !== requestScope) return false;
      fetchAutoTopUpConfig();
      return true;
    },
    [api, fetchAutoTopUpConfig, scopedIdentityKey, identityScope],
  );

  const requestTopUp = useCallback(
    async (credit_amount: number) => {
      if (scopedIdentityKey === null) {
        throw new Error("Authentication required");
      }
      const requestScope = identityScope;
      if (identityScopeRef.current.token !== requestScope) return;
      let response: { checkout_url: string };
      try {
        response = await api.requestTopUp(credit_amount);
      } catch (error) {
        if (identityScopeRef.current.token !== requestScope) return;
        throw error;
      }
      if (identityScopeRef.current.token !== requestScope) return;
      router.push(response.checkout_url);
    },
    [api, scopedIdentityKey, identityScope, router],
  );

  const openBillingPortal = useCallback(async () => {
    if (scopedIdentityKey === null) {
      throw new Error("Authentication required");
    }
    const requestScope = identityScope;
    if (identityScopeRef.current.token !== requestScope) return;
    let response: { url: string };
    try {
      response = await api.getUserPaymentPortalLink();
    } catch (error) {
      if (identityScopeRef.current.token !== requestScope) return;
      throw error;
    }
    if (identityScopeRef.current.token !== requestScope) return;
    router.push(response.url);
  }, [api, scopedIdentityKey, identityScope, router]);

  const refundTopUp = useCallback(
    async (transaction_key: string, reason: string) => {
      if (scopedIdentityKey === null) {
        throw new Error("Authentication required");
      }
      const requestScope = identityScope;
      if (identityScopeRef.current.token !== requestScope) return null;
      try {
        const refundedAmount = await api.refundTopUp(transaction_key, reason);
        if (identityScopeRef.current.token !== requestScope) return null;
        await fetchCredits();
        if (identityScopeRef.current.token !== requestScope) return null;
        const response = await api.getTransactionHistory();
        if (identityScopeRef.current.token !== requestScope) return null;
        setBillingState((current) =>
          current.identityScope === requestScope
            ? { ...current, transactionHistory: response }
            : current,
        );
        return refundedAmount;
      } catch (error) {
        if (identityScopeRef.current.token !== requestScope) return null;
        throw error;
      }
    },
    [api, fetchCredits, scopedIdentityKey, identityScope],
  );

  const fetchTransactionHistoryPage = useCallback(
    async (nextTransactionTime: Date | null, replace: boolean) => {
      const requestScope = identityScope;
      if (
        scopedIdentityKey === null ||
        identityScopeRef.current.token !== requestScope
      ) {
        return;
      }
      const response = await api.getTransactionHistory(nextTransactionTime, 20);
      setBillingState((current) =>
        current.identityScope === requestScope
          ? {
              ...current,
              transactionHistory: {
                transactions: replace
                  ? response.transactions
                  : [
                      ...current.transactionHistory.transactions,
                      ...response.transactions,
                    ],
                next_transaction_time: response.next_transaction_time,
              },
            }
          : current,
      );
    },
    [api, scopedIdentityKey, identityScope],
  );

  const fetchTransactionHistory = useCallback(async () => {
    await fetchTransactionHistoryPage(
      transactionHistory.next_transaction_time,
      false,
    );
  }, [fetchTransactionHistoryPage, transactionHistory.next_transaction_time]);

  useEffect(() => {
    if (!fetchInitialTransactionHistory) return;
    fetchTransactionHistoryPage(null, true);
  }, [fetchInitialTransactionHistory, fetchTransactionHistoryPage]);

  const fetchRefundRequests = useCallback(async () => {
    const requestScope = identityScope;
    if (
      scopedIdentityKey === null ||
      identityScopeRef.current.token !== requestScope
    ) {
      return;
    }
    const response = await api.getRefundRequests();
    setBillingState((current) =>
      current.identityScope === requestScope
        ? { ...current, refundRequests: response }
        : current,
    );
  }, [api, scopedIdentityKey, identityScope]);

  useEffect(() => {
    if (!fetchInitialRefundRequests) return;
    fetchRefundRequests();
  }, [fetchRefundRequests, fetchInitialRefundRequests]);

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
    openBillingPortal,
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
