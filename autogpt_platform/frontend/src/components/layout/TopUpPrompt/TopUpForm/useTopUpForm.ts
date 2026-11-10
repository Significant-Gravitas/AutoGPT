import { useToastOnFail } from "@/components/molecules/Toast/use-toast";
import { useCheckoutRedirect } from "@/hooks/useCheckoutRedirect";
import useCredits from "@/hooks/useCredits";
import { zodResolver } from "@hookform/resolvers/zod";
import { useState } from "react";
import { useForm } from "react-hook-form";
import { z } from "zod";

const topUpSchema = z.object({
  amount: z
    .number({ coerce: true, invalid_type_error: "Enter top-up amount" })
    .int("Enter a whole-dollar amount")
    .min(5, "Top-ups start at $5. Please enter a higher amount."),
});

export function useTopUpForm() {
  const toastOnFail = useToastOnFail();
  const { requestTopUp } = useCredits();

  const [isRequesting, setIsRequesting] = useState(false);
  const checkoutRedirect = useCheckoutRedirect();
  const isLoading = isRequesting || checkoutRedirect.isRedirectingHere;

  const form = useForm<z.infer<typeof topUpSchema>>({
    resolver: zodResolver(topUpSchema),
  });

  async function submitTopUp(data: z.infer<typeof topUpSchema>) {
    if (isLoading || checkoutRedirect.isRedirecting) return;
    setIsRequesting(true);
    await requestTopUp(Math.round(data.amount * 100))
      .then(() => checkoutRedirect.markRedirecting())
      .catch(toastOnFail("request top-up"))
      .finally(() => setIsRequesting(false));
  }

  return {
    form,
    isLoading,
    // Any checkout on the page leaving for Stripe, not just this one.
    isRedirecting: checkoutRedirect.isRedirecting,
    submitTopUp,
  };
}
