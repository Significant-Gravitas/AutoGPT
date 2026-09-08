import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { Form, FormField } from "@/components/molecules/Form/Form";
import useCredits from "@/hooks/useCredits";
import { useAuthStore } from "@/lib/auth/hooks/useAuthStore";
import { useTopUpForm } from "./useTopUpForm";

interface Props {
  submitLabel?: string;
  size?: "small" | "normal";
}

export function TopUpForm({ submitLabel = "Top up", size = "normal" }: Props) {
  const identityKey = useAuthStore((state) => state.user?.id ?? null);
  const { requestTopUp } = useCredits({ identityKey });

  return (
    <TopUpFormState
      key={identityKey ?? "logged-out"}
      requestTopUp={requestTopUp}
      size={size}
      submitLabel={submitLabel}
    />
  );
}

interface StateProps {
  requestTopUp: (creditAmount: number) => Promise<void>;
  submitLabel: string;
  size: "small" | "normal";
}

function TopUpFormState({ requestTopUp, submitLabel, size }: StateProps) {
  const { form, isLoading, submitTopUp } = useTopUpForm(requestTopUp);
  const inputSize = size === "small" ? "small" : "medium";
  const buttonSize = size === "small" ? "small" : "large";

  return (
    <Form form={form} onSubmit={submitTopUp} className="my-4">
      <FormField
        control={form.control}
        name="amount"
        render={({ field }) => (
          <Input
            label="Amount"
            type="amount"
            size={inputSize}
            decimalCount={0}
            id={field.name}
            error={form.formState.errors.amount?.message}
            amountPrefix="$"
            {...field}
          />
        )}
      />
      <Button type="submit" disabled={isLoading} size={buttonSize}>
        {isLoading ? "Redirecting…" : submitLabel}
      </Button>
    </Form>
  );
}
