import {
  useToast,
  useToastOnFail,
} from "@/components/molecules/Toast/use-toast";
import { zodResolver } from "@hookform/resolvers/zod";
import { useEffect, useState } from "react";
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

export function useAutoRefillForm(
  autoTopUpConfig: { amount: number; threshold: number } | null,
  updateAutoTopUpConfig: (
    amount: number,
    threshold: number,
  ) => Promise<boolean>,
) {
  const { toast } = useToast();
  const toastOnFail = useToastOnFail();
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

  async function submitAutoTopUpConfig(data: z.infer<typeof autoRefillSchema>) {
    setIsLoading(true);
    await updateAutoTopUpConfig(data.refillAmount * 100, data.threshold * 100)
      .then((updated) => {
        if (!updated) return;
        toast({
          title: "Auto top-up config updated! 🎉",
          variant: "success",
        });
      })
      .catch(toastOnFail("update auto top-up config"));
    setIsLoading(false);
  }

  return { autoRefillForm, isLoading, submitAutoTopUpConfig };
}
