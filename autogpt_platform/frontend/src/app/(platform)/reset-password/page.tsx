"use client";
import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { AuthCard } from "@/components/auth/AuthCard";
import { ExpiredLinkMessage } from "@/components/auth/ExpiredLinkMessage";
import { Form, FormField } from "@/components/__legacy__/ui/form";
import LoadingBox from "@/components/__legacy__/ui/loading";
import { useToast } from "@/components/molecules/Toast/use-toast";
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
  const [showExpiredMessage, setShowExpiredMessage] = useState(false);

  useEffect(() => {
    const error = searchParams.get("error");
    const errorCode = searchParams.get("error_code");
    const errorDescription = searchParams.get("error_description");

    if (error || errorCode) {
      // Check if this is an expired/used link error
      // Avoid broad checks like "invalid" which can match unrelated errors (e.g., PKCE errors)
      const descLower = errorDescription?.toLowerCase() || "";
      const isExpiredOrUsed =
        error === "link_expired" ||
        errorCode === "otp_expired" ||
        descLower.includes("expired") ||
        descLower.includes("already") ||
        descLower.includes("used");

      if (isExpiredOrUsed) {
        setShowExpiredMessage(true);
      } else {
        // Show toast for other errors
        const errorMessage =
          errorDescription || error || "Password reset failed";
        toast({
          title: "Password Reset Failed",
          description: errorMessage,
          variant: "destructive",
        });
      }

      // Clear all error params from URL
      const newUrl = new URL(window.location.href);
      newUrl.searchParams.delete("error");
      newUrl.searchParams.delete("error_code");
      newUrl.searchParams.delete("error_description");
      router.replace(newUrl.pathname + newUrl.search);
    }
  }, [searchParams, toast, router]);

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

      if (!(await sendEmailForm.trigger())) {
        setIsLoading(false);
        return;
      }

      const error = await sendResetEmail(data.email);
      setIsLoading(false);
      if (error) {
        toast({
          title: "Error",
          description: error,
          variant: "destructive",
        });
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
    [sendEmailForm, toast],
  );

  function handleShowEmailForm() {
    setShowExpiredMessage(false);
  }

  const onChangePassword = useCallback(
    async (data: z.infer<typeof changePasswordFormSchema>) => {
      setIsLoading(true);

      if (!(await changePasswordForm.trigger())) {
        setIsLoading(false);
        return;
      }

      const error = await changePassword(data.password);
      setIsLoading(false);
      if (error) {
        toast({
          title: "Error",
          description: error,
          variant: "destructive",
        });
        return;
      }
      toast({
        title: "Success",
        description: "Password changed successfully. Redirecting to login.",
        variant: "default",
      });
    },
    [changePasswordForm, toast],
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

  // Show expired link message if detected
  if (showExpiredMessage && !user) {
    return (
      <div className="flex h-full min-h-[85vh] w-full flex-col items-center justify-center">
        <AuthCard title="Reset Password">
          <ExpiredLinkMessage onRequestNewLink={handleShowEmailForm} />
        </AuthCard>
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
