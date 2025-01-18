"use client";
import { Button } from "@/components/agptui/Button";
import useCredits from "@/hooks/useCredits";
import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import { useSearchParams, useRouter } from "next/navigation";
import { useEffect, useState } from "react";

export default function CreditsPage() {
  const { requestTopUp } = useCredits();
  const [amount, setAmount] = useState(5);
  const [patched, setPatched] = useState(false);
  const searchParams = useSearchParams();
  const router = useRouter();
  const topupStatus = searchParams.get("topup");
  const api = useBackendAPI();

  useEffect(() => {
    if (!patched && topupStatus === "success") {
      api.fulfillCheckout();
      setPatched(true);
    }
  }, [api, patched, topupStatus]);

  const openBillingPortal = async () => {
    const portal = await api.getUserPaymentPortalLink();
    router.push(portal.url);
  };

  return (
    <div className="w-full min-w-[800px] px-4 sm:px-8">
      <h1 className="font-circular mb-6 text-[28px] font-normal text-neutral-900 dark:text-neutral-100 sm:mb-8 sm:text-[35px]">
        Credits
      </h1>

      <div className="grid grid-cols-1 gap-8 lg:grid-cols-2">
        {/* Left Column */}
        <div>
          <h2 className="text-lg">Top-up Credits</h2>

          <p className="mb-6 text-neutral-600 dark:text-neutral-400">
            {topupStatus === "success" && (
              <span className="text-green-500">
                Your payment was successful. Your credits will be updated
                shortly.
              </span>
            )}
            {topupStatus === "cancel" && (
              <span className="text-red-500">
                Payment failed. Your payment method has not been charged.
              </span>
            )}
          </p>

          <div className="mb-4 w-full">
            <label className="text-neutral-700">
              1 USD = 100 credits, 5 USD is a minimum top-up
            </label>
            <div className="rounded-[55px] border border-slate-200 px-4 py-2.5 dark:border-slate-700 dark:bg-slate-800">
              <input
                type="number"
                name="displayName"
                value={amount}
                placeholder="Top-up amount in USD"
                min="5"
                step="1"
                className="w-full"
                onChange={(e) => setAmount(parseInt(e.target.value))}
              />
            </div>
          </div>

          <Button
            type="submit"
            variant="default"
            className="font-circular ml-auto"
            onClick={() => requestTopUp(amount)}
          >
            Top-up
          </Button>
        </div>

        {/* Right Column */}
        <div>
          <h2 className="text-lg">Manage Your Payment Methods</h2>
          <br />
          <p className="text-neutral-600">
            You can manage your cards and see your payment history in the
            billing portal.
          </p>
          <br />

          <Button
            type="submit"
            variant="default"
            className="font-circular ml-auto"
            onClick={() => openBillingPortal()}
          >
            Open Portal
          </Button>
        </div>
      </div>
    </div>
  );
}
