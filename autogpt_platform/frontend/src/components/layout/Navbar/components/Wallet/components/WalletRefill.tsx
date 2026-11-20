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
import { TopUpForm } from "@/components/layout/TopUpPrompt/TopUpForm/TopUpForm";
import useCredits from "@/hooks/useCredits";
import { useAuthStore } from "@/lib/auth/hooks/useAuthStore";
import Link from "next/link";
import { FormProvider } from "react-hook-form";
import { useAutoRefillForm } from "./useAutoRefillForm";

export function WalletRefill() {
  const identityKey = useAuthStore((state) => state.user?.id ?? null);
  const { autoTopUpConfig, updateAutoTopUpConfig } = useCredits({
    identityKey,
    fetchInitialAutoTopUpConfig: identityKey !== null,
  });

  return (
    <WalletRefillState
      key={identityKey ?? "logged-out"}
      autoTopUpConfig={autoTopUpConfig}
      updateAutoTopUpConfig={updateAutoTopUpConfig}
    />
  );
}

interface Props {
  autoTopUpConfig: { amount: number; threshold: number } | null;
  updateAutoTopUpConfig: (
    amount: number,
    threshold: number,
  ) => Promise<boolean>;
}

function WalletRefillState({ autoTopUpConfig, updateAutoTopUpConfig }: Props) {
  const autoRefill = useAutoRefillForm(autoTopUpConfig, updateAutoTopUpConfig);

  return (
    <div className="mx-1 border-b border-border-strong">
      <Text
        variant="small-medium-tight"
        as="p"
        tone="accent"
        className="mx-0 mt-4"
      >
        Add credits to your balance
      </Text>
      <Text variant="small-tight" as="p" tone="muted" className="mx-0 my-1">
        Choose a one-time top-up or set up automatic refills
      </Text>
      <TabsLine
        variant="pill"
        defaultValue="top-up"
        className="mt-4 mb-6 flex w-full flex-col items-center"
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
            <AutoRefillForm {...autoRefill} />
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

function AutoRefillForm({
  autoRefillForm,
  isLoading,
  submitAutoTopUpConfig,
}: ReturnType<typeof useAutoRefillForm>) {
  return (
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
              error={autoRefillForm.formState.errors.refillAmount?.message}
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
  );
}
