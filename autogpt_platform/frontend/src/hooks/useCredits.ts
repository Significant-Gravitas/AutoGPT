import AutoGPTServerAPI, { RequestTopUpResponse } from "@/lib/autogpt-server-api";
import { useCallback, useEffect, useMemo, useState } from "react";
import { loadStripe } from "@stripe/stripe-js";
import { useRouter } from "next/navigation";

const stripePromise = loadStripe(process.env.NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY!);

export default function useCredits(): {
  credits: number | null;
  fetchCredits: () => void;
  requestTopUp: (amount: number) => Promise<void>;
} {
  const [credits, setCredits] = useState<number | null>(null);
  const api = useMemo(() => new AutoGPTServerAPI(), []);
  const router = useRouter();

  const fetchCredits = useCallback(async () => {
    const response = await api.getUserCredit();
    setCredits(response.credits);
  }, []);

  useEffect(() => {
    fetchCredits();
  }, [fetchCredits]);

  const requestTopUp = useCallback(async (amount: number) => {
    const stripe = await stripePromise;

    if (!stripe) {
      console.error("Stripe failed to load");
      return;
    }

    const response = await api.requestTopUp(amount);

    router.push(response.checkout_url);

    // const result = await stripe.confirmPayment({
    //   clientSecret: response.client_secret,
    //   confirmParams: {
    //     return_url: "return_to_url",
    //   },
    // });
  }, [api]);

  return {
    credits,
    fetchCredits,
    requestTopUp,
  };
}
