import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { cn } from "@/lib/utils";
import { zodResolver } from "@hookform/resolvers/zod";
import { useForm } from "react-hook-form";
import { z } from "zod";
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/ui/form";
import { Input } from "../ui/input";
import Link from "next/link";

const topUpSchema = z.object({
  amount: z
    .number({ coerce: true, invalid_type_error: "Enter top-up amount" })
    .min(5, "Top-ups start at $5. Please enter a higher amount."),
});

const autoRefillSchema = z
  .object({
    minBalance: z
      .number({ coerce: true, invalid_type_error: "Enter min. balance" })
      .min(
        5,
        "Looks like your balance is too low for auto-refill. Try $5 or more.",
      ),
    refillAmount: z
      .number({ coerce: true, invalid_type_error: "Enter top-up amount" })
      .min(5, "Top-ups start at $5. Please enter a higher amount."),
  })
  .refine((data) => data.minBalance < data.refillAmount, {
    message:
      "Your refill amount must be equal to or greater than the balance you entered above.",
    path: ["refillAmount"],
  });

export default function WalletRefill() {
  const topUpForm = useForm<z.infer<typeof topUpSchema>>({
    resolver: zodResolver(topUpSchema),
  });
  const autoRefillForm = useForm<z.infer<typeof autoRefillSchema>>({
    resolver: zodResolver(autoRefillSchema),
  });

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
              <form onSubmit={topUpForm.handleSubmit(() => {})}>
                <FormField
                  control={topUpForm.control}
                  name="amount"
                  render={({ field }) => (
                    <FormItem className="mb-6 mt-4">
                      <FormLabel className="font-sans text-sm font-medium leading-snug text-zinc-800">
                        Amount
                      </FormLabel>
                      <FormControl>
                        <>
                          <Input
                            className={cn(
                              "mt-2 rounded-3xl border-0 bg-white py-2 pl-6 pr-4 font-sans outline outline-1 outline-zinc-300",
                              "focus:outline-2 focus:outline-offset-0 focus:outline-violet-700",
                            )}
                            type="number"
                            step="1"
                            {...field}
                          />
                          <span className="absolute left-10 -translate-y-9 text-sm text-zinc-500">
                            $
                          </span>
                        </>
                      </FormControl>
                      <FormMessage className="mt-2 font-sans text-xs font-normal leading-tight" />
                    </FormItem>
                  )}
                />
                <button
                  className={cn(
                    "mb-2 inline-flex h-10 w-24 items-center justify-center rounded-3xl bg-zinc-800 px-4 py-2",
                    "font-sans text-sm font-medium leading-snug text-white",
                    "transition-colors duration-200 hover:bg-zinc-700",
                  )}
                  type="submit"
                >
                  Top up
                </button>
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
              <form onSubmit={autoRefillForm.handleSubmit(() => {})}>
                <FormField
                  control={autoRefillForm.control}
                  name="minBalance"
                  render={({ field }) => (
                    <FormItem className="mb-6 mt-4">
                      <FormLabel className="font-sans text-sm font-medium leading-snug text-zinc-800">
                        Refill when balance drops below:
                      </FormLabel>
                      <FormControl>
                        <>
                          <Input
                            className={cn(
                              "mt-2 rounded-3xl border-0 bg-white py-2 pl-6 pr-4 font-sans outline outline-1 outline-zinc-300",
                              "focus:outline-2 focus:outline-offset-0 focus:outline-violet-700",
                            )}
                            type="number"
                            step="1"
                            {...field}
                          />
                          <span className="absolute left-10 -translate-y-9 text-sm text-zinc-500">
                            $
                          </span>
                        </>
                      </FormControl>
                      <FormMessage className="mt-2 font-sans text-xs font-normal leading-tight" />
                    </FormItem>
                  )}
                />
                <FormField
                  control={autoRefillForm.control}
                  name="refillAmount"
                  render={({ field }) => (
                    <FormItem className="mb-6">
                      <FormLabel className="font-sans text-sm font-medium leading-snug text-zinc-800">
                        Add this amount:
                      </FormLabel>
                      <FormControl>
                        <>
                          <Input
                            className={cn(
                              "mt-2 rounded-3xl border-0 bg-white py-2 pl-6 pr-4 font-sans outline outline-1 outline-zinc-300",
                              "focus:outline-2 focus:outline-offset-0 focus:outline-violet-700",
                            )}
                            type="number"
                            step="1"
                            {...field}
                          />
                          <span className="absolute left-10 -translate-y-9 text-sm text-zinc-500">
                            $
                          </span>
                        </>
                      </FormControl>
                      <FormMessage className="mt-2 font-sans text-xs font-normal leading-tight" />
                    </FormItem>
                  )}
                />
                <button
                  className={cn(
                    "mb-4 inline-flex h-10 w-40 items-center justify-center rounded-3xl bg-zinc-800 px-4 py-2",
                    "font-sans text-sm font-medium leading-snug text-white",
                    "transition-colors duration-200 hover:bg-zinc-700",
                  )}
                  type="submit"
                >
                  Enable Auto-refill
                </button>
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
