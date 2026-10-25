import { Form, FormField } from "@/components/__legacy__/ui/form";
import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import {
  TabsLine,
  TabsLineContent,
  TabsLineList,
  TabsLineTrigger,
} from "@/components/molecules/TabsLine/TabsLine";
import {
  useToast,
  useToastOnFail,
} from "@/components/molecules/Toast/use-toast";
import { TopUpForm } from "@/components/layout/TopUpPrompt/TopUpForm/TopUpForm";
import useCredits from "@/hooks/useCredits";
import { zodResolver } from "@hookform/resolvers/zod";
import Link from "next/link";
import { useCallback, useEffect, useState } from "react";
import { useForm } from "react-hook-form";
import { z } from "zod";

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

  const { autoTopUpConfig, updateAutoTopUpConfig } = useCredits({
    fetchInitialAutoTopUpConfig: true,
  });

  const [isLoading, setIsLoading] = useState(false);

  const autoRefillForm = useForm<z.infer<typeof autoRefillSchema>>({
    resolver: zodResolver(autoRefillSchema),
  });

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
    <div className="mx-1 border-b border-border-strong">
      <p className="mx-0 mt-4 font-sans text-xs font-medium text-accent-strong">
        Add credits to your balance
      </p>
      <p className="mx-0 my-1 font-sans text-xs font-normal text-foreground-tertiary">
        Choose a one-time top-up or set up automatic refills
      </p>
      <TabsLine
        variant="pill"
        defaultValue="top-up"
        className="mb-6 mt-4 flex w-full flex-col items-center"
      >
        <TabsLineList className="mx-auto">
          <TabsLineTrigger value="top-up">One-time top up</TabsLineTrigger>
          <TabsLineTrigger value="auto-refill">Auto-refill</TabsLineTrigger>
        </TabsLineList>
        <div className="mt-4 w-full rounded-lg px-5 outline outline-1 outline-offset-2 outline-border-base">
          <TabsLineContent value="top-up" className="flex flex-col">
            <div className="mt-2 justify-start font-sans text-sm font-medium leading-snug text-foreground-strong">
              One-time top-up
            </div>
            <div className="mt-1 justify-start font-sans text-xs font-normal leading-tight text-foreground-tertiary">
              Enter an amount (min. $5) and add credits instantly.
            </div>
            <TopUpForm submitLabel="Top up" size="small" />
          </TabsLineContent>
          <TabsLineContent value="auto-refill" className="flex flex-col">
            <div className="justify-start font-sans text-sm font-medium leading-snug text-foreground-strong">
              Auto-refill
            </div>
            <div className="mt-1 justify-start font-sans text-xs font-normal leading-tight text-foreground-tertiary">
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
          </TabsLineContent>
          <div className="mb-3 justify-start font-sans text-xs font-normal leading-tight">
            <span className="text-foreground-tertiary">
              To update your billing details, head to{" "}
            </span>
            <Link
              href="/settings/billing"
              className="cursor-pointer text-foreground-emphasis underline"
            >
              Billing settings
            </Link>
          </div>
        </div>
      </TabsLine>
    </div>
  );
}
