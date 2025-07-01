"use client";
import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { AuthCard } from "@/components/auth/AuthCard";
import AuthFeedback from "@/components/auth/AuthFeedback";
import { PasswordInput } from "@/components/auth/PasswordInput";
import Turnstile from "@/components/auth/Turnstile";
import {
  Form,
  FormControl,
  FormDescription,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/ui/form";
import LoadingBox from "@/components/ui/loading";
import { useTurnstile } from "@/hooks/useTurnstile";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import { getBehaveAs } from "@/lib/utils";
import { changePasswordFormSchema, sendEmailFormSchema } from "@/types/auth";
import { zodResolver } from "@hookform/resolvers/zod";
import { useCallback, useState } from "react";
import { useForm } from "react-hook-form";
import { z } from "zod";
import { changePassword, sendResetEmail } from "./actions";

export default function ResetPasswordPage() {
  const { supabase, user, isUserLoading } = useSupabase();
  const [isLoading, setIsLoading] = useState(false);
  const [feedback, setFeedback] = useState<string | null>(null);
  const [isError, setIsError] = useState(false);
  const [disabled, setDisabled] = useState(false);
  const [sendEmailCaptchaKey, setSendEmailCaptchaKey] = useState(0);
  const [changePasswordCaptchaKey, setChangePasswordCaptchaKey] = useState(0);

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

  const resetSendEmailCaptcha = useCallback(() => {
    setSendEmailCaptchaKey((k) => k + 1);
    sendEmailTurnstile.reset();
  }, [sendEmailTurnstile]);

  const resetChangePasswordCaptcha = useCallback(() => {
    setChangePasswordCaptchaKey((k) => k + 1);
    changePasswordTurnstile.reset();
  }, [changePasswordTurnstile]);

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
        resetSendEmailCaptcha();
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
        resetSendEmailCaptcha();
        return;
      }
      setDisabled(true);
      setFeedback(
        "Password reset email sent if user exists. Please check your email.",
      );
      setIsError(false);
    },
    [sendEmailForm, sendEmailTurnstile, resetSendEmailCaptcha],
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
        resetChangePasswordCaptcha();
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
        resetChangePasswordCaptcha();
        return;
      }
      setFeedback("Password changed successfully. Redirecting to login.");
      setIsError(false);
    },
    [changePasswordForm, changePasswordTurnstile, resetChangePasswordCaptcha],
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
    <div className="flex h-full min-h-[85vh] flex-col items-center justify-center">
      <AuthCard title="Reset Password">
        {user ? (
          <form
            onSubmit={changePasswordForm.handleSubmit(onChangePassword)}
            className="flex w-full flex-col gap-1"
          >
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
                      Password needs to be at least 12 characters long
                    </FormDescription>
                    <FormMessage />
                  </FormItem>
                )}
              />

              {/* Turnstile CAPTCHA Component for password change */}
              <Turnstile
                key={changePasswordCaptchaKey}
                siteKey={changePasswordTurnstile.siteKey}
                onVerify={changePasswordTurnstile.handleVerify}
                onExpire={changePasswordTurnstile.handleExpire}
                onError={changePasswordTurnstile.handleError}
                setWidgetId={changePasswordTurnstile.setWidgetId}
                action="change_password"
                shouldRender={changePasswordTurnstile.shouldRender}
              />

              <Button
                variant="primary"
                loading={isLoading}
                type="submit"
                className="mt-6 w-full"
                onClick={() => onChangePassword(changePasswordForm.getValues())}
              >
                Update password
              </Button>
              <AuthFeedback
                type="login"
                message={feedback}
                isError={isError}
                behaveAs={getBehaveAs()}
              />
            </Form>
          </form>
        ) : (
          <form
            onSubmit={sendEmailForm.handleSubmit(onSendEmail)}
            className="flex w-full flex-col gap-1"
          >
            <Form {...sendEmailForm}>
              <FormField
                control={sendEmailForm.control}
                name="email"
                render={({ field }) => (
                  <FormItem className="mb-6">
                    <FormControl>
                      <Input
                        label="Email"
                        placeholder="m@example.com"
                        type="email"
                        error={sendEmailForm.formState.errors.email?.message}
                        {...field}
                      />
                    </FormControl>
                  </FormItem>
                )}
              />

              {/* Turnstile CAPTCHA Component for reset email */}
              <Turnstile
                key={sendEmailCaptchaKey}
                siteKey={sendEmailTurnstile.siteKey}
                onVerify={sendEmailTurnstile.handleVerify}
                onExpire={sendEmailTurnstile.handleExpire}
                onError={sendEmailTurnstile.handleError}
                setWidgetId={sendEmailTurnstile.setWidgetId}
                action="reset_password"
                shouldRender={sendEmailTurnstile.shouldRender}
              />

              <Button
                variant="primary"
                loading={isLoading}
                disabled={disabled}
                type="submit"
                className="mt-6 w-full"
                onClick={() => onSendEmail(sendEmailForm.getValues())}
              >
                Send reset email
              </Button>
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
