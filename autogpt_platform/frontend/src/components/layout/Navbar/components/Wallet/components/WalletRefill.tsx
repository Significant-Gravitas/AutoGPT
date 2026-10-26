import { FormField } from "@/components/molecules/Form/Form";
import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { Text } from "@/components/atoms/Text/Text";
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
import { useForm, FormProvider } from "react-hook-form";
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
      <Text
        variant="small-medium-tight"
        as="p"
        className="mx-0 mt-4 text-accent-strong"
      >
        Add credits to your balance
      </Text>
      <Text variant="small-tight" as="p" tone="muted" className="mx-0 my-1">
        Choose a one-time top-up or set up automatic refills
      </Text>
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
            <Text
              variant="body-medium-lh-snug"
              as="div"
              tone="primary"
              className="mt-2 justify-start"
            >
              One-time top-up
            </Text>
            <Text
              variant="small-lh-tight"
              as="div"
              tone="muted"
              className="mt-1 justify-start"
            >
              Enter an amount (min. $5) and add credits instantly.
            </Text>
            <TopUpForm submitLabel="Top up" size="small" />
          </TabsLineContent>
          <TabsLineContent value="auto-refill" className="flex flex-col">
            <Text
              variant="body-medium-lh-snug"
              as="div"
              tone="primary"
              className="justify-start"
            >
              Auto-refill
            </Text>
            <Text
              variant="small-lh-tight"
              as="div"
              tone="muted"
              className="mt-1 justify-start"
            >
              Choose a one-time top-up or set up automatic refills.
            </Text>

            <FormProvider {...autoRefillForm}>
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
            </FormProvider>
          </TabsLineContent>
          <Text
            variant="small-lh-tight"
            as="div"
            tone="inherit"
            className="mb-3 justify-start"
          >
            <span className="text-foreground-tertiary">
              To update your billing details, head to{" "}
            </span>
            <Link
              href="/settings/billing"
              className="cursor-pointer text-foreground-emphasis underline"
            >
              Billing settings
            </Link>
          </Text>
        </div>
      </TabsLine>
    </div>
  );
}
