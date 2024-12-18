"use client";
import { Button } from "@/components/ui/button";
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import useSupabase from "@/hooks/useSupabase";
import { zodResolver } from "@hookform/resolvers/zod";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { useForm } from "react-hook-form";
import { FaSpinner } from "react-icons/fa";
import { z } from "zod";

const emailFormSchema = z.object({
  email: z.string().email().min(2).max(64),
});

const resetPasswordFormSchema = z
  .object({
    password: z.string().min(6).max(64),
    confirmPassword: z.string().min(6).max(64),
  })
  .refine((data) => data.password === data.confirmPassword, {
    message: "Passwords don't match",
    path: ["confirmPassword"],
  });

export default function ResetPasswordPage() {
  const { supabase, user, isUserLoading } = useSupabase();
  const router = useRouter();
  const [isLoading, setIsLoading] = useState(false);
  const [feedback, setFeedback] = useState<string | null>(null);

  const emailForm = useForm<z.infer<typeof emailFormSchema>>({
    resolver: zodResolver(emailFormSchema),
    defaultValues: {
      email: "",
    },
  });

  const resetPasswordForm = useForm<z.infer<typeof resetPasswordFormSchema>>({
    resolver: zodResolver(resetPasswordFormSchema),
    defaultValues: {
      password: "",
      confirmPassword: "",
    },
  });

  if (isUserLoading) {
    return (
      <div className="flex h-[80vh] items-center justify-center">
        <FaSpinner className="mr-2 h-16 w-16 animate-spin" />
      </div>
    );
  }

  if (!supabase) {
    return (
      <div>
        User accounts are disabled because Supabase client is unavailable
      </div>
    );
  }

  async function onSendEmail(d: z.infer<typeof emailFormSchema>) {
    setIsLoading(true);
    setFeedback(null);

    if (!(await emailForm.trigger())) {
      setIsLoading(false);
      return;
    }

    const { data, error } = await supabase!.auth.resetPasswordForEmail(
      d.email,
      {
        redirectTo: `${window.location.origin}/reset_password`,
      },
    );

    if (error) {
      setFeedback(error.message);
      setIsLoading(false);
      return;
    }

    setFeedback("Password reset email sent. Please check your email.");
    setIsLoading(false);
  }

  async function onResetPassword(d: z.infer<typeof resetPasswordFormSchema>) {
    setIsLoading(true);
    setFeedback(null);

    if (!(await resetPasswordForm.trigger())) {
      setIsLoading(false);
      return;
    }

    const { data, error } = await supabase!.auth.updateUser({
      password: d.password,
    });

    if (error) {
      setFeedback(error.message);
      setIsLoading(false);
      return;
    }

    await supabase!.auth.signOut();
    router.push("/login");
  }

  return (
    <div className="flex h-full flex-col items-center justify-center">
      <div className="w-full max-w-md">
        <h1 className="text-center text-3xl font-bold">Reset Password</h1>
        {user ? (
          <form
            onSubmit={resetPasswordForm.handleSubmit(onResetPassword)}
            className="mt-6 space-y-6"
          >
            <Form {...resetPasswordForm}>
              <FormField
                control={resetPasswordForm.control}
                name="password"
                render={({ field }) => (
                  <FormItem className="mb-4">
                    <FormLabel>Password</FormLabel>
                    <FormControl>
                      <Input
                        type="password"
                        placeholder="password"
                        {...field}
                      />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
              <FormField
                control={resetPasswordForm.control}
                name="confirmPassword"
                render={({ field }) => (
                  <FormItem className="mb">
                    <FormLabel>Confirm Password</FormLabel>
                    <FormControl>
                      <Input
                        type="password"
                        placeholder="password"
                        {...field}
                      />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
              <Button
                type="submit"
                className="w-full"
                disabled={isLoading}
                onClick={() => onResetPassword(resetPasswordForm.getValues())}
              >
                {isLoading ? <FaSpinner className="mr-2 animate-spin" /> : null}
                Reset Password
              </Button>
            </Form>
          </form>
        ) : (
          <form
            onSubmit={emailForm.handleSubmit(onSendEmail)}
            className="mt-6 space-y-6"
          >
            <Form {...emailForm}>
              <FormField
                control={emailForm.control}
                name="email"
                render={({ field }) => (
                  <FormItem className="mb-4">
                    <FormLabel>Email</FormLabel>
                    <FormControl>
                      <Input placeholder="user@email.com" {...field} />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
              <Button
                type="submit"
                className="w-full"
                disabled={isLoading}
                onClick={() => onSendEmail(emailForm.getValues())}
              >
                {isLoading ? <FaSpinner className="mr-2 animate-spin" /> : null}
                Send Reset Email
              </Button>
              {feedback ? (
                <div className="text-center text-sm text-red-500">
                  {feedback}
                </div>
              ) : null}
            </Form>
          </form>
        )}
      </div>
    </div>
  );
}
