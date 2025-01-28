import AutoGPTServerAPI from "@/lib/autogpt-server-api";
import { useCallback, useEffect, useMemo, useState } from "react";
import { loadStripe } from "@stripe/stripe-js";
import { useRouter } from "next/navigation";

const stripePromise = loadStripe(
  process.env.NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY!,
);

export default function useCredits(): {
  credits: number | null;
  fetchCredits: () => void;
  requestTopUp: (amount: number) => Promise<void>;
  autoTopUpConfig: { amount: number; threshold: number } | null;
  fetchAutoTopUpConfig: () => void;
  updateAutoTopUpConfig: (amount: number, threshold: number) => Promise<void>;
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
    async (amount: number) => {
      const stripe = await stripePromise;

      if (!stripe) {
        return;
      }

      // Convert dollar amount to credit count
      const response = await api.requestTopUp(amount);
      router.push(response.checkout_url);
    },
    [api, router],
  );

  return {
    credits,
    fetchCredits,
    requestTopUp,
    autoTopUpConfig,
    fetchAutoTopUpConfig,
    updateAutoTopUpConfig,
  };
}
