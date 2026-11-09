import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { Form, FormField } from "@/components/molecules/Form/Form";
import { useTopUpForm } from "./useTopUpForm";

interface Props {
  submitLabel?: string;
  size?: "small" | "normal";
  onRequestingChange?: (isRequesting: boolean) => void;
  onRequestFailed?: () => void;
}

export function TopUpForm({
  submitLabel = "Top up",
  size = "normal",
  onRequestingChange,
  onRequestFailed,
}: Props) {
  const { form, isLoading, isRedirecting, submitTopUp } = useTopUpForm({
    onRequestingChange,
    onRequestFailed,
  });
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
      <Button
        type="submit"
        disabled={isLoading || isRedirecting}
        size={buttonSize}
      >
        {isLoading ? "Redirecting…" : submitLabel}
      </Button>
    </Form>
  );
}
