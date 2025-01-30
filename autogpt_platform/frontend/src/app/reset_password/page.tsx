"use client";
import {
  AuthCard,
  AuthHeader,
  AuthButton,
  AuthFeedback,
  PasswordInput,
} from "@/components/auth";
import {
  Form,
  FormControl,
  FormDescription,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import useSupabase from "@/hooks/useSupabase";
import { sendEmailFormSchema, changePasswordFormSchema } from "@/types/auth";
import { zodResolver } from "@hookform/resolvers/zod";
import { useCallback, useState } from "react";
import { useForm } from "react-hook-form";
import { z } from "zod";
import { changePassword, sendResetEmail } from "./actions";
import Spinner from "@/components/Spinner";

export default function ResetPasswordPage() {
  const { supabase, user, isUserLoading } = useSupabase();
  const [isLoading, setIsLoading] = useState(false);
  const [feedback, setFeedback] = useState<string | null>(null);
  const [isError, setIsError] = useState(false);
  const [disabled, setDisabled] = useState(false);

  const sendEmailForm = useForm<z.infer<typeof sendEmailFormSchema>>({
    resolver: zodResolver(sendEmailFormSchema),
    defaultValues: {
      email: "",
    },
  });

  const changePasswordForm = useForm<z.infer<typeof changePasswordFormSchema>>({
    resolver: zodResolver(changePasswordFormSchema),
    defaultValues: {
      password: "",
      confirmPassword: "",
    },
  });

  const onSendEmail = useCallback(
    async (data: z.infer<typeof sendEmailFormSchema>) => {
      setIsLoading(true);
      setFeedback(null);

      if (!(await sendEmailForm.trigger())) {
        setIsLoading(false);
        return;
      }

      const error = await sendResetEmail(data.email);
      setIsLoading(false);
      if (error) {
        setFeedback(error);
        setIsError(true);
        return;
      }
      setDisabled(true);
      setFeedback(
        "Password reset email sent if user exists. Please check your email.",
      );
      setIsError(false);
    },
    [sendEmailForm],
  );

  const onChangePassword = useCallback(
    async (data: z.infer<typeof changePasswordFormSchema>) => {
      setIsLoading(true);
      setFeedback(null);

      if (!(await changePasswordForm.trigger())) {
        setIsLoading(false);
        return;
      }

      const error = await changePassword(data.password);
      setIsLoading(false);
      if (error) {
        setFeedback(error);
        setIsError(true);
        return;
      }
      setFeedback("Password changed successfully. Redirecting to login.");
      setIsError(false);
    },
    [changePasswordForm],
  );

  if (isUserLoading) {
    return <Spinner />;
  }

  if (!supabase) {
    return (
      <div>
        User accounts are disabled because Supabase client is unavailable
      </div>
    );
  }

  return (
    <div className="flex min-h-screen items-center justify-center">
      <AuthCard>
        <AuthHeader>Reset Password</AuthHeader>
        {user ? (
          <form onSubmit={changePasswordForm.handleSubmit(onChangePassword)}>
            <Form {...changePasswordForm}>
              <FormField
                control={changePasswordForm.control}
                name="password"
                render={({ field }) => (
                  <FormItem className="mb-6">
                    <FormLabel>Password</FormLabel>
                    <FormControl>
                      <PasswordInput {...field} />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
              <FormField
                control={changePasswordForm.control}
                name="confirmPassword"
                render={({ field }) => (
                  <FormItem className="mb-6">
                    <FormLabel>Confirm Password</FormLabel>
                    <FormControl>
                      <PasswordInput {...field} />
                    </FormControl>
                    <FormDescription className="text-sm font-normal leading-tight text-slate-500">
                      Password needs to be at least 6 characters long
                    </FormDescription>
                    <FormMessage />
                  </FormItem>
                )}
              />
              <AuthButton
                onClick={() => onChangePassword(changePasswordForm.getValues())}
                isLoading={isLoading}
                type="submit"
              >
                Update password
              </AuthButton>
              <AuthFeedback message={feedback} isError={isError} />
            </Form>
          </form>
        ) : (
          <form onSubmit={sendEmailForm.handleSubmit(onSendEmail)}>
            <Form {...sendEmailForm}>
              <FormField
                control={sendEmailForm.control}
                name="email"
                render={({ field }) => (
                  <FormItem className="mb-6">
                    <FormLabel>Email</FormLabel>
                    <FormControl>
                      <Input placeholder="m@example.com" {...field} />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
              <AuthButton
                onClick={() => onSendEmail(sendEmailForm.getValues())}
                isLoading={isLoading}
                disabled={disabled}
                type="submit"
              >
                Send reset email
              </AuthButton>
              <AuthFeedback message={feedback} isError={isError} />
            </Form>
          </form>
        )}
      </AuthCard>
    </div>
  );
}
