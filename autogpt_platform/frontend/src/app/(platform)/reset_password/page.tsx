"use client";
import {
  AuthCard,
  AuthHeader,
  AuthButton,
  AuthFeedback,
  PasswordInput,
  Turnstile,
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
import LoadingBox from "@/components/ui/loading";
import { getBehaveAs } from "@/lib/utils";
import { useTurnstile } from "@/hooks/useTurnstile";

export default function ResetPasswordPage() {
  const { supabase, user, isUserLoading } = useSupabase();
  const [isLoading, setIsLoading] = useState(false);
  const [feedback, setFeedback] = useState<string | null>(null);
  const [isError, setIsError] = useState(false);
  const [disabled, setDisabled] = useState(false);

  const sendEmailTurnstile = useTurnstile({
    action: "reset_password",
    autoVerify: false,
    resetOnError: true,
  });

  const changePasswordTurnstile = useTurnstile({
    action: "change_password",
    autoVerify: false,
    resetOnError: true,
  });

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

      if (!sendEmailTurnstile.verified) {
        setFeedback("Please complete the CAPTCHA challenge.");
        setIsError(true);
        setIsLoading(false);
        return;
      }

      const error = await sendResetEmail(
        data.email,
        sendEmailTurnstile.token as string,
      );
      setIsLoading(false);
      if (error) {
        setFeedback(error);
        setIsError(true);
        sendEmailTurnstile.reset();
        return;
      }
      setDisabled(true);
      setFeedback(
        "Password reset email sent if user exists. Please check your email.",
      );
      setIsError(false);
    },
    [sendEmailForm, sendEmailTurnstile],
  );

  const onChangePassword = useCallback(
    async (data: z.infer<typeof changePasswordFormSchema>) => {
      setIsLoading(true);
      setFeedback(null);

      if (!(await changePasswordForm.trigger())) {
        setIsLoading(false);
        return;
      }

      if (!changePasswordTurnstile.verified) {
        setFeedback("Please complete the CAPTCHA challenge.");
        setIsError(true);
        setIsLoading(false);
        return;
      }

      const error = await changePassword(
        data.password,
        changePasswordTurnstile.token as string,
      );
      setIsLoading(false);
      if (error) {
        setFeedback(error);
        setIsError(true);
        changePasswordTurnstile.reset();
        return;
      }
      setFeedback("Password changed successfully. Redirecting to login.");
      setIsError(false);
    },
    [changePasswordForm, changePasswordTurnstile],
  );

  if (isUserLoading) {
    return <LoadingBox className="h-[80vh]" />;
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

              {/* Turnstile CAPTCHA Component for password change */}
              <Turnstile
                siteKey={changePasswordTurnstile.siteKey}
                onVerify={changePasswordTurnstile.handleVerify}
                onExpire={changePasswordTurnstile.handleExpire}
                onError={changePasswordTurnstile.handleError}
                action="change_password"
                shouldRender={changePasswordTurnstile.shouldRender}
              />

              <AuthButton
                onClick={() => onChangePassword(changePasswordForm.getValues())}
                isLoading={isLoading}
                type="submit"
              >
                Update password
              </AuthButton>
              <AuthFeedback
                type="login"
                message={feedback}
                isError={isError}
                behaveAs={getBehaveAs()}
              />
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

              {/* Turnstile CAPTCHA Component for reset email */}
              <Turnstile
                siteKey={sendEmailTurnstile.siteKey}
                onVerify={sendEmailTurnstile.handleVerify}
                onExpire={sendEmailTurnstile.handleExpire}
                onError={sendEmailTurnstile.handleError}
                action="reset_password"
                shouldRender={sendEmailTurnstile.shouldRender}
              />

              <AuthButton
                onClick={() => onSendEmail(sendEmailForm.getValues())}
                isLoading={isLoading}
                disabled={disabled}
                type="submit"
              >
                Send reset email
              </AuthButton>
              <AuthFeedback
                type="login"
                message={feedback}
                isError={isError}
                behaveAs={getBehaveAs()}
              />
            </Form>
          </form>
        )}
      </AuthCard>
    </div>
  );
}
