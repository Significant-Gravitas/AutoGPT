"use client";
import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { AuthCard } from "@/components/auth/AuthCard";
import Turnstile from "@/components/auth/Turnstile";
import { Form, FormField } from "@/components/ui/form";
import LoadingBox from "@/components/ui/loading";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { useTurnstile } from "@/hooks/useTurnstile";
import { useSupabase } from "@/lib/supabase/hooks/useSupabase";
import { changePasswordFormSchema, sendEmailFormSchema } from "@/types/auth";
import { zodResolver } from "@hookform/resolvers/zod";
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useCallback, useEffect, useState } from "react";
import { useForm } from "react-hook-form";
import { z } from "zod";
import { changePassword, sendResetEmail } from "./actions";

function ResetPasswordContent() {
  const { supabase, user, isUserLoading } = useSupabase();
  const { toast } = useToast();
  const searchParams = useSearchParams();
  const router = useRouter();
  const [isLoading, setIsLoading] = useState(false);
  const [disabled, setDisabled] = useState(false);
  const [sendEmailCaptchaKey, setSendEmailCaptchaKey] = useState(0);
  const [changePasswordCaptchaKey, setChangePasswordCaptchaKey] = useState(0);

  useEffect(() => {
    const error = searchParams.get("error");
    if (error) {
      toast({
        title: "Password Reset Failed",
        description: error,
        variant: "destructive",
      });

      const newUrl = new URL(window.location.href);
      newUrl.searchParams.delete("error");
      router.replace(newUrl.pathname + newUrl.search);
    }
  }, [searchParams, toast, router]);

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

      if (!(await sendEmailForm.trigger())) {
        setIsLoading(false);
        return;
      }

      if (!sendEmailTurnstile.verified) {
        toast({
          title: "CAPTCHA Required",
          description: "Please complete the CAPTCHA challenge.",
          variant: "destructive",
        });
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
        toast({
          title: "Error",
          description: error,
          variant: "destructive",
        });
        resetSendEmailCaptcha();
        return;
      }
      setDisabled(true);
      toast({
        title: "Email Sent",
        description:
          "Password reset email sent if user exists. Please check your email.",
        variant: "default",
      });
    },
    [sendEmailForm, sendEmailTurnstile, resetSendEmailCaptcha, toast],
  );

  const onChangePassword = useCallback(
    async (data: z.infer<typeof changePasswordFormSchema>) => {
      setIsLoading(true);

      if (!(await changePasswordForm.trigger())) {
        setIsLoading(false);
        return;
      }

      if (!changePasswordTurnstile.verified) {
        toast({
          title: "CAPTCHA Required",
          description: "Please complete the CAPTCHA challenge.",
          variant: "destructive",
        });
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
        toast({
          title: "Error",
          description: error,
          variant: "destructive",
        });
        resetChangePasswordCaptcha();
        return;
      }
      toast({
        title: "Success",
        description: "Password changed successfully. Redirecting to login.",
        variant: "default",
      });
    },
    [
      changePasswordForm,
      changePasswordTurnstile,
      resetChangePasswordCaptcha,
      toast,
    ],
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
    <div className="flex h-full min-h-[85vh] w-full flex-col items-center justify-center">
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
                  <Input
                    id={field.name}
                    label="Password"
                    type="password"
                    placeholder="••••••••••••••••"
                    error={
                      changePasswordForm.formState.errors.password?.message
                    }
                    {...field}
                  />
                )}
              />
              <FormField
                control={changePasswordForm.control}
                name="confirmPassword"
                render={({ field }) => (
                  <Input
                    id={field.name}
                    label="Confirm Password"
                    type="password"
                    placeholder="••••••••••••••••"
                    error={
                      changePasswordForm.formState.errors.confirmPassword
                        ?.message
                    }
                    {...field}
                  />
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
                {isLoading ? "Updating password..." : "Update password"}
              </Button>
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
                  <Input
                    id={field.name}
                    label="Email"
                    placeholder="m@example.com"
                    type="email"
                    error={sendEmailForm.formState.errors.email?.message}
                    {...field}
                  />
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
            </Form>
          </form>
        )}
      </AuthCard>
    </div>
  );
}

export default function ResetPasswordPage() {
  return (
    <Suspense fallback={<LoadingBox className="h-[80vh]" />}>
      <ResetPasswordContent />
    </Suspense>
  );
}
