import { Form, FormField } from "@/components/__legacy__/ui/form";
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/components/__legacy__/ui/tabs";
import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import {
  useToast,
  useToastOnFail,
} from "@/components/molecules/Toast/use-toast";
import useCredits from "@/hooks/useCredits";
import { zodResolver } from "@hookform/resolvers/zod";
import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import { useForm } from "react-hook-form";
import { z } from "zod";

const topUpSchema = z.object({
  amount: z
    .number({ coerce: true, invalid_type_error: "Enter top-up amount" })
    .min(5, "Top-ups start at $5. Please enter a higher amount."),
});

const autoRefillSchema = z
  .object({
    threshold: z
      .number({ coerce: true, invalid_type_error: "Enter min. balance" })
      .min(
        5,
        "Looks like your balance is too low for auto-refill. Try $5 or more.",
      ),
    refillAmount: z
      .number({ coerce: true, invalid_type_error: "Enter top-up amount" })
      .min(5, "Top-ups start at $5. Please enter a higher amount."),
  })
  .refine((data) => data.refillAmount >= data.threshold, {
    message:
      "Your refill amount must be equal to or greater than the balance you entered above.",
    path: ["refillAmount"],
  });

export function WalletRefill() {
  const { toast } = useToast();
  const toastOnFail = useToastOnFail();

  const { requestTopUp, autoTopUpConfig, updateAutoTopUpConfig } = useCredits({
    fetchInitialAutoTopUpConfig: true,
  });

  const [isLoading, setIsLoading] = useState(false);

  const topUpForm = useForm<z.infer<typeof topUpSchema>>({
    resolver: zodResolver(topUpSchema),
  });
  const autoRefillForm = useForm<z.infer<typeof autoRefillSchema>>({
    resolver: zodResolver(autoRefillSchema),
  });

  console.log("autoRefillForm");

  // Pre-fill the auto-refill form with existing values
  useEffect(() => {
    if (
      autoTopUpConfig &&
      autoTopUpConfig.amount > 0 &&
      autoTopUpConfig.threshold > 0 &&
      !autoRefillForm.getFieldState("threshold").isTouched &&
      !autoRefillForm.getFieldState("refillAmount").isTouched
    ) {
      autoRefillForm.setValue("threshold", autoTopUpConfig.threshold / 100);
      autoRefillForm.setValue("refillAmount", autoTopUpConfig.amount / 100);
    }
  }, [autoTopUpConfig, autoRefillForm]);

  const submitTopUp = useCallback(
    async (data: z.infer<typeof topUpSchema>) => {
      setIsLoading(true);
      await requestTopUp(data.amount * 100).catch(
        toastOnFail("request top-up"),
      );
      setIsLoading(false);
    },
    [requestTopUp, toastOnFail],
  );

  const submitAutoTopUpConfig = useCallback(
    async (data: z.infer<typeof autoRefillSchema>) => {
      setIsLoading(true);
      await updateAutoTopUpConfig(data.refillAmount * 100, data.threshold * 100)
        .then(() => {
          toast({
            title: "Auto top-up config updated! 🎉",
            variant: "success",
          });
        })
        .catch(toastOnFail("update auto top-up config"));
      setIsLoading(false);
    },
    [updateAutoTopUpConfig, toast, toastOnFail],
  );

  return (
    <div className="mx-1 border-b border-zinc-300">
      <p className="mx-0 mt-4 font-sans text-xs font-medium text-violet-700">
        Add credits to your balance
      </p>
      <p className="mx-0 my-1 font-sans text-xs font-normal text-zinc-500">
        Choose a one-time top-up or set up automatic refills
      </p>
      <Tabs
        defaultValue="top-up"
        className="mb-6 mt-4 flex w-full flex-col items-center"
      >
        <TabsList className="mx-auto">
          <TabsTrigger value="top-up">One-time top up</TabsTrigger>
          <TabsTrigger value="auto-refill">Auto-refill</TabsTrigger>
        </TabsList>
        <div className="mt-4 w-full rounded-lg px-5 outline outline-1 outline-offset-2 outline-zinc-200">
          <TabsContent value="top-up" className="flex flex-col">
            <div className="mt-2 justify-start font-sans text-sm font-medium leading-snug text-zinc-900">
              One-time top-up
            </div>
            <div className="mt-1 justify-start font-sans text-xs font-normal leading-tight text-zinc-500">
              Enter an amount (min. $5) and add credits instantly.
            </div>
            <Form {...topUpForm}>
              <form
                onSubmit={topUpForm.handleSubmit(submitTopUp)}
                className="my-4"
              >
                <FormField
                  control={topUpForm.control}
                  name="amount"
                  render={({ field }) => (
                    <Input
                      label="Amount"
                      type="amount"
                      size="small"
                      decimalCount={0}
                      id={field.name}
                      error={topUpForm.formState.errors.amount?.message}
                      amountPrefix="$"
                      {...field}
                    />
                  )}
                />
                <Button type="submit" disabled={isLoading} size="small">
                  Top up
                </Button>
              </form>
            </Form>
          </TabsContent>
          <TabsContent value="auto-refill" className="flex flex-col">
            <div className="justify-start font-sans text-sm font-medium leading-snug text-zinc-900">
              Auto-refill
            </div>
            <div className="mt-1 justify-start font-sans text-xs font-normal leading-tight text-zinc-500">
              Choose a one-time top-up or set up automatic refills.
            </div>

            <Form {...autoRefillForm}>
              <form
                onSubmit={autoRefillForm.handleSubmit(submitAutoTopUpConfig)}
                className="my-6"
              >
                <FormField
                  control={autoRefillForm.control}
                  name="threshold"
                  render={({ field }) => (
                    <Input
                      type="amount"
                      label="Refill when balance drops below:"
                      id={field.name}
                      size="small"
                      decimalCount={0}
                      error={autoRefillForm.formState.errors.threshold?.message}
                      amountPrefix="$"
                      {...field}
                    />
                  )}
                />
                <FormField
                  control={autoRefillForm.control}
                  name="refillAmount"
                  render={({ field }) => (
                    <Input
                      type="amount"
                      label="Add this amount:"
                      size="small"
                      decimalCount={0}
                      id={field.name}
                      error={
                        autoRefillForm.formState.errors.refillAmount?.message
                      }
                      amountPrefix="$"
                      {...field}
                    />
                  )}
                />
                <Button
                  type="submit"
                  disabled={isLoading}
                  size="small"
                  className="mt-5"
                >
                  Enable Auto-refill
                </Button>
              </form>
            </Form>
          </TabsContent>
          <div className="mb-3 justify-start font-sans text-xs font-normal leading-tight">
            <span className="text-zinc-500">
              To update your billing details, head to{" "}
            </span>
            <Link
              href="/profile/credits"
              className="cursor-pointer text-zinc-800 underline"
            >
              Billing settings
            </Link>
          </div>
        </div>
      </Tabs>
    </div>
  );
}
