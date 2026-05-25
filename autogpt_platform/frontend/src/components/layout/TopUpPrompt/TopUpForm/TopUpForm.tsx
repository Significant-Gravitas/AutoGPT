import { Form, FormField } from "@/components/__legacy__/ui/form";
import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { useTopUpForm } from "./useTopUpForm";

interface Props {
  submitLabel?: string;
}

export function TopUpForm({ submitLabel = "Top up" }: Props) {
  const { form, isLoading, submitTopUp } = useTopUpForm();

  return (
    <Form {...form}>
      <form onSubmit={form.handleSubmit(submitTopUp)} className="my-4">
        <FormField
          control={form.control}
          name="amount"
          render={({ field }) => (
            <Input
              label="Amount"
              type="amount"
              size="small"
              decimalCount={0}
              id={field.name}
              error={form.formState.errors.amount?.message}
              amountPrefix="$"
              {...field}
            />
          )}
        />
        <Button type="submit" disabled={isLoading} size="small">
          {submitLabel}
        </Button>
      </form>
    </Form>
  );
}
