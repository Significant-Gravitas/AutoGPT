"use client";
import { useEffect, useState } from "react";
import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { Text } from "@/components/atoms/Text/Text";
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
} from "@/components/molecules/Table/TablePrimitives";

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
      <Text variant="h5" as="h3">
        Expert Usage & Storage
      </Text>
      <div className="flex flex-col gap-3 rounded-lg border border-zinc-200 p-4">
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
        size="md"
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
    const form = e.currentTarget;
    const amount =
      parseInt(new FormData(form).get("topUpAmount") as string) * 100;
    requestTopUp(amount).catch(toastOnFail("request top-up"));
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
      <Text variant="h2" as="h1" tone="primary" className="mb-6 sm:mb-8">
        Billing
      </Text>

      {/* Subscription Tier */}
      <div className="mb-8">
        <SubscriptionTierSection />
      </div>

      <div className="grid grid-cols-1 gap-8 lg:grid-cols-2">
        {/* Top-up Form */}
        <div className="space-y-4">
          <Text variant="h5" as="h3">
            Top-up Credits
          </Text>

          <Text variant="large" tone="secondary" className="mb-6">
            {topupStatus === "success" && (
              <span className="text-green-500">
                Your payment was successful. Your credits will be updated
                shortly. Try refreshing the page in case it is not updated.
              </span>
            )}
            {topupStatus === "cancel" && (
              <span className="text-red-500">
                Payment failed. Your payment method has not been charged.
              </span>
            )}
          </Text>

          <form onSubmit={submitTopUp} className="space-y-4">
            <Input
              type="number"
              id="topUpAmount"
              name="topUpAmount"
              label="Top-up amount (USD), minimum $5:"
              labelVariant="large"
              labelClassName="text-zinc-700"
              placeholder="Enter top-up amount"
              min="5"
              step="1"
              defaultValue={5}
              required
            />

            <Button type="submit" size="md" className="w-full">
              Top-up
            </Button>
          </form>

          {/* Auto Top-up Form */}
          <form onSubmit={submitAutoTopUpConfig} className="my-6 space-y-4">
            <Text variant="h5" as="h3">
              Automatic Refill Settings
            </Text>

            <Input
              type="number"
              id="threshold"
              name="threshold"
              label="When my balance goes below this amount:"
              labelVariant="large"
              labelClassName="text-zinc-700"
              defaultValue={
                autoTopUpConfig?.threshold
                  ? autoTopUpConfig.threshold / 100
                  : ""
              }
              placeholder="Refill threshold, minimum $5"
              min="5"
              step="1"
              required
            />

            <Input
              type="number"
              id="autoTopUpAmount"
              name="topUpAmount"
              label="Automatically refill my balance with this amount:"
              labelVariant="large"
              labelClassName="text-zinc-700"
              defaultValue={
                autoTopUpConfig?.amount ? autoTopUpConfig.amount / 100 : ""
              }
              placeholder="Refill amount, minimum $5"
              min="5"
              step="1"
              required
            />

            <Text variant="body">
              <b>Note:</b> For your safety, we will top up your balance{" "}
              <b>at most once</b> per agent execution to prevent unintended
              excessive charges. Therefore, ensure that the automatic top-up
              amount is sufficient for your agent&apos;s operation.
            </Text>

            {autoTopUpConfig?.amount ? (
              <>
                <Button type="submit" size="md" className="w-full">
                  Save Changes
                </Button>
                <Button
                  size="md"
                  className="w-full"
                  variant="destructive"
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
              <Button type="submit" size="md" className="w-full">
                Enable Auto-Refill
              </Button>
            )}
          </form>

          {/* Expert Usage Limits */}
          <CoPilotUsageSection />
        </div>

        <div className="my-6 space-y-4">
          {/* Payment Portal */}
          <Text variant="h5" as="h3">
            Manage Your Payment Methods
          </Text>
          <Text variant="large" tone="secondary">
            You can manage your cards and see your payment history in the
            billing portal.
          </Text>
          <Button
            size="md"
            type="submit"
            className="w-full"
            onClick={() => openBillingPortal()}
          >
            Open Portal
          </Button>

          {/* Transaction History */}
          <Text variant="h5" as="h3">
            Transaction History
          </Text>
          {transactionHistory.transactions.length === 0 && (
            <Text variant="large" tone="secondary">
              No transactions found.
            </Text>
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
                      transaction.amount > 0 ? "text-green-500" : "text-red-500"
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
              size="md"
              type="submit"
              className="w-full"
              onClick={() => fetchTransactionHistory()}
            >
              Load More
            </Button>
          )}

          {refundRequests.length > 0 && (
            <>
              <Text variant="h5" as="h3">
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
            size="md"
            variant="destructive"
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
