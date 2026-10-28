"use client";
import { useEffect, useState } from "react";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { useCheckoutRedirect } from "@/hooks/useCheckoutRedirect";
import useCredits from "@/hooks/useCredits";
import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import { useSearchParams, useRouter } from "next/navigation";
import {
  useToast,
  useToastOnFail,
} from "@/components/molecules/Toast/use-toast";

import { RefundModal } from "./RefundModal";
import { SubscriptionTierSection } from "./components/SubscriptionTierSection/SubscriptionTierSection";
import { CreditTransaction } from "@/lib/autogpt-server-api";
import { StorageBar } from "@/app/(platform)/copilot/components/UsageLimits/StorageBar";
import { UsageBar } from "@/app/(platform)/copilot/components/UsageLimits/UsageBar";
import type { CoPilotUsagePublic } from "@/app/api/__generated__/models/coPilotUsagePublic";
import { useGetV2GetCopilotUsage } from "@/app/api/__generated__/endpoints/chat/chat";

import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/molecules/Table/components/TablePrimitives";

function CoPilotUsageSection() {
  const router = useRouter();
  const { data: usage, isSuccess } = useGetV2GetCopilotUsage({
    query: {
      select: (res) => res.data as CoPilotUsagePublic,
      refetchInterval: 30000,
      staleTime: 10000,
    },
  });

  if (!isSuccess || !usage) return null;
  if (!usage.daily && !usage.weekly) return null;

  return (
    <div className="my-6 space-y-4">
      <Text variant="lg-medium" as="h3" tone="inherit">
        Expert Usage & Storage
      </Text>
      <div
        // TODO(tokens): no exact role (border-neutral-200 #E5E5E5)
        className="flex flex-col gap-3 rounded-lg border border-neutral-200 p-4"
      >
        {usage.daily && (
          <UsageBar
            label="Today"
            percentUsed={usage.daily.percent_used}
            resetsAt={usage.daily.resets_at}
          />
        )}
        {usage.weekly && (
          <UsageBar
            label="This week"
            percentUsed={usage.weekly.percent_used}
            resetsAt={usage.weekly.resets_at}
          />
        )}
        <StorageBar />
      </div>
      <Button
        variant="legacy-default"
        className="w-full"
        onClick={() => router.push("/copilot")}
      >
        Open Otto
      </Button>
    </div>
  );
}

export default function CreditsPage() {
  const api = useBackendAPI();
  const {
    requestTopUp,
    autoTopUpConfig,
    updateAutoTopUpConfig,
    transactionHistory,
    fetchTransactionHistory,
    formatCredits,
    refundTopUp,
    refundRequests,
  } = useCredits({
    fetchInitialAutoTopUpConfig: true,
    fetchInitialRefundRequests: true,
    fetchInitialTransactionHistory: true,
  });
  const router = useRouter();
  const searchParams = useSearchParams();
  const topupStatus = searchParams.get("topup") as "success" | "cancel" | null;
  const { toast } = useToast();
  const toastOnFail = useToastOnFail();
  const checkoutRedirect = useCheckoutRedirect();
  const [isRequestingTopUp, setIsRequestingTopUp] = useState(false);

  const [isRefundModalOpen, setIsRefundModalOpen] = useState(false);
  const [topUpTransactions, setTopUpTransactions] = useState<
    CreditTransaction[]
  >([]);
  const openRefundModal = () => {
    api.getTransactionHistory(null, 20, "TOP_UP").then((history) => {
      setTopUpTransactions(history.transactions);
      setIsRefundModalOpen(true);
    });
  };
  const refundCredits = (transaction_key: string, reason: string) =>
    refundTopUp(transaction_key, reason)
      .then((amount) => {
        if (amount > 0) {
          toast({
            title: "Refund approved! 🎉",
            description: `Your refund has been automatically processed. Based on your remaining balance, the amount of ${formatCredits(amount)} will be credited to your account.`,
          });
        } else {
          toast({
            title: "Refund Request Received",
            description:
              "We have received your refund request. A member of our team will review it and reach out via email shortly.",
          });
        }
      })
      .catch(toastOnFail("refund transaction"));

  useEffect(() => {
    if (api && topupStatus === "success") {
      api.fulfillCheckout().catch(toastOnFail("fulfill checkout"));
    }
  }, [api, topupStatus, toastOnFail]);

  const openBillingPortal = () =>
    api
      .getUserPaymentPortalLink()
      .then((portal) => {
        router.push(portal.url);
      })
      .catch(toastOnFail("open billing portal"));

  const submitTopUp = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (isRequestingTopUp || checkoutRedirect.isRedirecting) return;
    const form = e.currentTarget;
    const amount =
      parseInt(new FormData(form).get("topUpAmount") as string) * 100;
    setIsRequestingTopUp(true);
    requestTopUp(amount)
      .then(() => checkoutRedirect.markRedirecting())
      .catch(toastOnFail("request top-up"))
      .finally(() => setIsRequestingTopUp(false));
  };

  const submitAutoTopUpConfig = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const form = e.currentTarget;
    const formData = new FormData(form);
    const amount = parseInt(formData.get("topUpAmount") as string) * 100;
    const threshold = parseInt(formData.get("threshold") as string) * 100;
    updateAutoTopUpConfig(amount, threshold)
      .then(() => {
        toast({ title: "Auto top-up config updated! 🎉" });
      })
      .catch(toastOnFail("update auto top-up config"));
  };

  return (
    <div className="w-full px-4 sm:px-8 md:min-w-[800px]">
      <h1
        // TODO(tokens): no exact role (text-neutral-900 #171717, dark:text-neutral-100); SECRT-3152 rare size (sm:text-[35px])
        className="mb-6 text-heading font-normal leading-inherit text-neutral-900 dark:text-neutral-100 sm:mb-8 sm:text-[35px]"
      >
        Billing
      </h1>

      {/* Subscription Tier */}
      <div className="mb-8">
        <SubscriptionTierSection />
      </div>

      <div className="grid grid-cols-1 gap-8 lg:grid-cols-2">
        {/* Top-up Form */}
        <div className="space-y-4">
          <Text variant="lg-medium" as="h3" tone="inherit">
            Top-up Credits
          </Text>

          <p
            // TODO(tokens): no exact role (text-neutral-600 #525252, dark:text-neutral-400)
            className="mb-6 text-neutral-600 dark:text-neutral-400"
          >
            {topupStatus === "success" && (
              <span
                // TODO(tokens): no exact role (text-green-500 #16A34A)
                className="text-green-500"
              >
                Your payment was successful. Your credits will be updated
                shortly. Try refreshing the page in case it is not updated.
              </span>
            )}
            {topupStatus === "cancel" && (
              <span className="text-error">
                Payment failed. Your payment method has not been charged.
              </span>
            )}
          </p>

          <form onSubmit={submitTopUp} className="space-y-4">
            <div>
              <label
                htmlFor="topUpAmount"
                // TODO(tokens): no exact role (text-neutral-700 #404040)
                className="mb-1 block text-neutral-700"
              >
                Top-up amount (USD), minimum $5:
              </label>
              <input
                type="number"
                id="topUpAmount"
                name="topUpAmount"
                placeholder="Enter top-up amount"
                min="5"
                step="1"
                defaultValue={5}
                // TODO(tokens): no exact role (border-slate-200 #CFD4DB, dark:border-slate-700, dark:bg-slate-800)
                className="w-full rounded-md border border-slate-200 px-4 py-2 dark:border-slate-700 dark:bg-slate-800"
                required
              />
            </div>

            <Button
              variant="legacy-default"
              type="submit"
              className="w-full"
              disabled={isRequestingTopUp || checkoutRedirect.isRedirecting}
            >
              Top-up
            </Button>
          </form>

          {/* Auto Top-up Form */}
          <form onSubmit={submitAutoTopUpConfig} className="my-6 space-y-4">
            <Text variant="lg-medium" as="h3" tone="inherit">
              Automatic Refill Settings
            </Text>

            <div>
              <label
                htmlFor="threshold"
                // TODO(tokens): no exact role (text-neutral-700 #404040)
                className="mb-1 block text-neutral-700"
              >
                When my balance goes below this amount:
              </label>
              <input
                type="number"
                id="threshold"
                name="threshold"
                defaultValue={
                  autoTopUpConfig?.threshold
                    ? autoTopUpConfig.threshold / 100
                    : ""
                }
                placeholder="Refill threshold, minimum $5"
                min="5"
                step="1"
                // TODO(tokens): no exact role (border-slate-200 #CFD4DB, dark:border-slate-700, dark:bg-slate-800)
                className="w-full rounded-md border border-slate-200 px-4 py-2 dark:border-slate-700 dark:bg-slate-800"
                required
              />
            </div>

            <div>
              <label
                htmlFor="autoTopUpAmount"
                // TODO(tokens): no exact role (text-neutral-700 #404040)
                className="mb-1 block text-neutral-700"
              >
                Automatically refill my balance with this amount:
              </label>
              <input
                type="number"
                id="autoTopUpAmount"
                name="topUpAmount"
                defaultValue={
                  autoTopUpConfig?.amount ? autoTopUpConfig.amount / 100 : ""
                }
                placeholder="Refill amount, minimum $5"
                min="5"
                step="1"
                // TODO(tokens): no exact role (border-slate-200 #CFD4DB, dark:border-slate-700, dark:bg-slate-800)
                className="w-full rounded-md border border-slate-200 px-4 py-2 dark:border-slate-700 dark:bg-slate-800"
                required
              />
            </div>

            <Text variant="body-tight" tone="inherit">
              <b>Note:</b> For your safety, we will top up your balance{" "}
              <b>at most once</b> per workflow execution to prevent unintended
              excessive charges. Therefore, ensure that the automatic top-up
              amount is sufficient for your workflow&apos;s operation.
            </Text>

            {autoTopUpConfig?.amount ? (
              <>
                <Button
                  variant="legacy-default"
                  type="submit"
                  className="w-full"
                >
                  Save Changes
                </Button>
                <Button
                  className="w-full"
                  variant="legacy-destructive"
                  onClick={() =>
                    updateAutoTopUpConfig(0, 0).then(() => {
                      toast({ title: "Auto top-up config disabled! 🎉" });
                    })
                  }
                >
                  Disable Auto-Refill
                </Button>
              </>
            ) : (
              <Button variant="legacy-default" type="submit" className="w-full">
                Enable Auto-Refill
              </Button>
            )}
          </form>

          {/* Expert Usage Limits */}
          <CoPilotUsageSection />
        </div>

        <div className="my-6 space-y-4">
          {/* Payment Portal */}

          <Text variant="lg-medium" as="h3" tone="inherit">
            Manage Your Payment Methods
          </Text>
          <p
            // TODO(tokens): no exact role (text-neutral-600 #525252)
            className="text-neutral-600"
          >
            You can manage your cards and see your payment history in the
            billing portal.
          </p>
          <Button
            variant="legacy-default"
            type="submit"
            className="w-full"
            onClick={() => openBillingPortal()}
          >
            Open Portal
          </Button>

          {/* Transaction History */}

          <Text variant="lg-medium" as="h3" tone="inherit">
            Transaction History
          </Text>
          {transactionHistory.transactions.length === 0 && (
            <p
              // TODO(tokens): no exact role (text-neutral-600 #525252)
              className="text-neutral-600"
            >
              No transactions found.
            </p>
          )}
          <Table
            className={
              transactionHistory.transactions.length === 0 ? "hidden" : ""
            }
          >
            <TableHeader>
              <TableRow>
                <TableHead>Date</TableHead>
                <TableHead>Description</TableHead>
                <TableHead>Amount</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {transactionHistory.transactions.map((transaction, i) => (
                <TableRow key={i}>
                  <TableCell>
                    {new Date(transaction.transaction_time).toLocaleString(
                      undefined,
                      {
                        month: "short",
                        day: "numeric",
                        year: "numeric",
                        hour: "numeric",
                        minute: "numeric",
                      },
                    )}
                  </TableCell>
                  <TableCell>{transaction.description}</TableCell>
                  {/* Make it green if it's positive, red if it's negative */}
                  <TableCell
                    className={
                      // TODO(tokens): no exact role (text-green-500 #16A34A)
                      transaction.amount > 0 ? "text-green-500" : "text-error"
                    }
                  >
                    <b>{formatCredits(transaction.amount)}</b>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
          {transactionHistory.next_transaction_time && (
            <Button
              variant="legacy-default"
              type="submit"
              className="w-full"
              onClick={() => fetchTransactionHistory()}
            >
              Load More
            </Button>
          )}

          {refundRequests.length > 0 && (
            <>
              <Text variant="lg-medium" as="h3" tone="inherit">
                Your Refund Requests
              </Text>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Last Updated</TableHead>
                    <TableHead>Amount</TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead>Comment</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {refundRequests.map((request, i) => (
                    <TableRow key={i}>
                      <TableCell>
                        {new Date(request.updated_at).toLocaleString(
                          undefined,
                          {
                            month: "short",
                            day: "numeric",
                            year: "numeric",
                            hour: "numeric",
                            minute: "numeric",
                          },
                        )}
                      </TableCell>
                      <TableCell>{formatCredits(request.amount)}</TableCell>
                      <TableCell>{request.status}</TableCell>
                      <TableCell>{request.result}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </>
          )}

          <Button
            variant="legacy-destructive"
            onClick={() => openRefundModal()}
            className="w-full"
          >
            Request Refund
          </Button>
          <RefundModal
            isOpen={isRefundModalOpen}
            onClose={() => setIsRefundModalOpen(false)}
            transactions={topUpTransactions}
            formatCredits={formatCredits}
            refundCredits={refundCredits}
          />
        </div>
      </div>
    </div>
  );
}
