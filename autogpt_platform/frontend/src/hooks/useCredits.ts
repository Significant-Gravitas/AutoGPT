import AutoGPTServerAPI from "@/lib/autogpt-server-api";
import {
  RefundRequest,
  TransactionHistory,
} from "@/lib/autogpt-server-api/types";
import { useActiveOrgWallet } from "@/services/org-team/useActiveOrgWallet";
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
  isCreditsForbidden: boolean;
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
  // A team organization has its own wallet: pin every request to it and keep
  // its balance apart from the personal one. Personal organizations keep the
  // client and identity they always had.
  const wallet = useActiveOrgWallet();
  const pinnedOrgID = wallet.isTeamOrg ? wallet.organizationID : null;
  const scopedIdentityKey = pinnedOrgID
    ? `${identityKey ?? ""}::${pinnedOrgID}`
    : identityKey;

  const [credits, setCredits] = useState<number | null>(null);
  const [creditsForbidden, setCreditsForbidden] = useState(false);
  const [autoTopUpConfig, setAutoTopUpConfig] = useState<{
    amount: number;
    threshold: number;
  } | null>(null);
  const identityKeyRef = useRef(scopedIdentityKey);
  const [stateIdentityKey, setStateIdentityKey] = useState(scopedIdentityKey);

  useEffect(() => {
    identityKeyRef.current = scopedIdentityKey;
    setStateIdentityKey(scopedIdentityKey);
    setCredits(null);
    setCreditsForbidden(false);
    setAutoTopUpConfig(null);
  }, [scopedIdentityKey]);

  const api = useMemo(() => {
    const client = new AutoGPTServerAPI();
    return pinnedOrgID
      ? client.withTenantScope({ organizationId: pinnedOrgID, teamId: null })
      : client;
  }, [pinnedOrgID]);
  const router = useRouter();

  const fetchCredits = useCallback(async () => {
    const requestIdentityKey = scopedIdentityKey;
    const response = await api.getUserCredit();
    if (identityKeyRef.current !== requestIdentityKey) return;
    setCredits(response.credits ?? null);
    setCreditsForbidden(Boolean(response.forbidden));
  }, [api, scopedIdentityKey]);

  useEffect(() => {
    if (!fetchInitialCredits) return;
    fetchCredits();
  }, [fetchCredits, fetchInitialCredits]);

  const fetchAutoTopUpConfig = useCallback(async () => {
    const requestIdentityKey = scopedIdentityKey;
    const response = await api.getAutoTopUpConfig();
    if (identityKeyRef.current !== requestIdentityKey) return;
    setAutoTopUpConfig(response);
  }, [api, scopedIdentityKey]);

  useEffect(() => {
    if (!fetchInitialAutoTopUpConfig) return;
    fetchAutoTopUpConfig();
  }, [fetchAutoTopUpConfig, fetchInitialAutoTopUpConfig]);

  const updateAutoTopUpConfig = useCallback(
    async (amount: number, threshold: number) => {
      await api.setAutoTopUpConfig({ amount, threshold });
      fetchAutoTopUpConfig();
    },
    [api, fetchAutoTopUpConfig],
  );

  const requestTopUp = useCallback(
    async (credit_amount: number) => {
      const response = await api.requestTopUp(credit_amount);
      if (!response.checkout_url) {
        throw new Error("Checkout didn't start. Please try again later.");
      }
      router.push(response.checkout_url);
    },
    [api, router],
  );

  const refundTopUp = useCallback(
    async (transaction_key: string, reason: string) => {
      const refunded_amount = await api.refundTopUp(transaction_key, reason);
      await fetchCredits();
      setTransactionHistory(await api.getTransactionHistory());
      return refunded_amount;
    },
    [api, fetchCredits],
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
    if (!fetchInitialTransactionHistory) return;
    fetchTransactionHistory();
    // Note: We only need to fetch transaction history once.
    // Hence, we should avoid `fetchTransactionHistory` to the dependency array.
  }, [fetchInitialTransactionHistory]);

  const [refundRequests, setRefundRequests] = useState<RefundRequest[]>([]);

  const fetchRefundRequests = useCallback(async () => {
    const response = await api.getRefundRequests();
    setRefundRequests(response);
  }, [api]);

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

  const isCurrentIdentity = stateIdentityKey === scopedIdentityKey;

  return {
    credits: isCurrentIdentity ? credits : null,
    isCreditsForbidden:
      isCurrentIdentity && pinnedOrgID !== null && creditsForbidden,
    fetchCredits,
    requestTopUp,
    refundTopUp,
    autoTopUpConfig: isCurrentIdentity ? autoTopUpConfig : null,
    fetchAutoTopUpConfig,
    updateAutoTopUpConfig,
    transactionHistory,
    fetchTransactionHistory,
    refundRequests,
    fetchRefundRequests,
    formatCredits,
  };
}
