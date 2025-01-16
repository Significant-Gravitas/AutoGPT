"use client";
import { Button } from "@/components/agptui/Button";
import useCredits from "@/hooks/useCredits";
import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import { useSearchParams } from "next/navigation";
import { useEffect, useState } from "react";

export default function CreditsPage() {
  const { credits, requestTopUp } = useCredits();
  const [amount, setAmount] = useState(5);
  const [patched, setPatched] = useState(false);
  const searchParams = useSearchParams();
  const topupStatus = searchParams.get("topup");
  const api = useBackendAPI();

  useEffect(() => {
    if (!patched && topupStatus === "success") {
      api.fulfillCheckout();
      setPatched(true);
    }
  }, [api, patched, topupStatus]);

  return (
    <div className="w-full min-w-[800px] px-4 sm:px-8">
      <h1 className="font-circular mb-6 text-[28px] font-normal text-neutral-900 dark:text-neutral-100 sm:mb-8 sm:text-[35px]">
        Credits
      </h1>
      <p className="font-circular mb-6 text-base font-normal leading-tight text-neutral-600 dark:text-neutral-400">
        Current credits: <b>{credits}</b>
      </p>
      <h2 className="font-circular mb-4 text-lg font-normal leading-7 text-neutral-700 dark:text-neutral-300">
        Top-up Credits
      </h2>
      <p className="font-circular mb-6 text-base font-normal leading-tight text-neutral-600 dark:text-neutral-400">
        {topupStatus === "success" && (
          <span className="text-green-500">
            Your payment was successful. Your credits will be updated shortly.
          </span>
        )}
        {topupStatus === "cancel" && (
          <span className="text-red-500">
            Payment failed. Your payment method has not been charged.
          </span>
        )}
      </p>
      <div className="w-full">
        <label className="font-circular mb-1.5 block text-base font-normal leading-tight text-neutral-700 dark:text-neutral-300">
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
            className="font-circular w-full border-none bg-transparent text-base font-normal text-neutral-900 placeholder:text-neutral-400 focus:outline-none dark:text-white dark:placeholder:text-neutral-500"
            onChange={(e) => setAmount(parseInt(e.target.value))}
          />
        </div>
      </div>
      <Button
        type="submit"
        variant="default"
        className="font-circular mt-4 h-[50px] rounded-[35px] bg-neutral-800 px-6 py-3 text-base font-medium text-white transition-colors hover:bg-neutral-900 dark:bg-neutral-200 dark:text-neutral-900 dark:hover:bg-neutral-100"
        onClick={() => requestTopUp(amount)}
      >
        {"Top-up"}
      </Button>
    </div>
  );
}
