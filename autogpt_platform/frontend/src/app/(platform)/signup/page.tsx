"use client";

import { Checkbox } from "@/components/__legacy__/ui/checkbox";
import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { Link } from "@/components/atoms/Link/Link";
import { Text } from "@/components/atoms/Text/Text";
import { AuthCard } from "@/components/auth/AuthCard";
import AuthFeedback from "@/components/auth/AuthFeedback";
import { EmailNotAllowedModal } from "@/components/auth/EmailNotAllowedModal";
import { GoogleOAuthButton } from "@/components/auth/GoogleOAuthButton";
import { MobileWarningBanner } from "@/components/auth/MobileWarningBanner";
import { environment } from "@/services/environment";
import { WarningOctagon } from "@phosphor-icons/react";
import { Controller, FormProvider } from "react-hook-form";
import { LoadingSignup } from "./components/LoadingSignup";
import { useSignupPage } from "./useSignupPage";

export default function SignupPage() {
  const {
    form,
    feedback,
    isLoggedIn,
    isLoading,
    isGoogleLoading,
    isCloudEnv,
    isUserLoading,
    showNotAllowedModal,
    isAuthAvailable,
    handleSubmit,
    handleProviderSignup,
    handleCloseNotAllowedModal,
  } = useSignupPage();

  if (isUserLoading || isLoggedIn) {
    return <LoadingSignup />;
  }

  if (!isAuthAvailable) {
    return <div>User accounts are disabled because auth is unavailable</div>;
  }

  const confirmPasswordError = form.formState.errors.confirmPassword?.message;
  const termsError = form.formState.errors.agreeToTerms?.message;

  return (
    <div className="flex h-full min-h-[85vh] flex-col items-center justify-center py-10">
      <AuthCard title="Create a new account">
        <FormProvider {...form}>
          <form onSubmit={handleSubmit} className="flex w-full flex-col gap-1">
            <Controller
              control={form.control}
              name="email"
              render={({ field }) => (
                <Input
                  id={field.name}
                  label="Email"
                  placeholder="m@example.com"
                  type="email"
                  autoComplete="email"
                  error={form.formState.errors.email?.message}
                  {...field}
                />
              )}
            />
            <Controller
              control={form.control}
              name="password"
              render={({ field }) => {
                return (
                  <Input
                    id={field.name}
                    label="Password"
                    placeholder="•••••••••••••••••••••"
                    type="password"
                    autoComplete="new-password"
                    error={form.formState.errors.password?.message}
                    {...field}
                  />
                );
              }}
            />
            <Controller
              control={form.control}
              name="confirmPassword"
              render={({ field }) => (
                <Input
                  id={field.name}
                  label="Confirm Password"
                  placeholder="•••••••••••••••••••••"
                  type="password"
                  autoComplete="new-password"
                  error={confirmPasswordError}
                  {...field}
                />
              )}
            />
            <Controller
              control={form.control}
              name="agreeToTerms"
              render={({ field }) => (
                <>
                  <div className="mt-6 flex w-full flex-row items-center space-x-2">
                    <Checkbox
                      id="agreeToTerms"
                      checked={field.value}
                      onCheckedChange={field.onChange}
                      className="relative bottom-px"
                    />
                    <label
                      htmlFor="agreeToTerms"
                      className="flex flex-wrap items-center gap-1"
                    >
                      <Text
                        variant="body-medium"
                        className="inline-block text-slate-950"
                      >
                        I agree to the
                      </Text>
                      <Link
                        href="https://auto-gpt.notion.site/Terms-of-Use-11400ef5bece80d0b087d7831c5fd6bf"
                        variant="secondary"
                      >
                        Terms of Use
                      </Link>
                      <Text
                        variant="body-medium"
                        className="inline-block text-slate-950"
                      >
                        and
                      </Text>
                      <Link
                        href="https://www.notion.so/auto-gpt/Privacy-Policy-ab11c9c20dbd4de1a15dcffe84d77984"
                        variant="secondary"
                      >
                        Privacy Policy
                      </Link>
                    </label>
                  </div>
                  {termsError ? (
                    <div className="flex items-center gap-2">
                      <WarningOctagon className="h-4 w-4 text-red-500" />
                      <Text variant="small-medium" className="!text-red-500">
                        {termsError}
                      </Text>
                    </div>
                  ) : null}
                </>
              )}
            />

            <Button
              variant="primary"
              loading={isLoading}
              disabled={isGoogleLoading}
              type="submit"
              className="mt-6 w-full"
            >
              {isLoading ? "Signing up..." : "Sign up"}
            </Button>
          </form>
        </FormProvider>
        {isCloudEnv ? (
          <GoogleOAuthButton
            onClick={() => handleProviderSignup("google")}
            isLoading={isGoogleLoading}
            disabled={isLoading}
          />
        ) : null}
        <AuthFeedback
          type="signup"
          message={feedback}
          isError={!!feedback}
          behaveAs={environment.getBehaveAs()}
        />

        <AuthCard.BottomText
          text="Already a member?"
          link={{ text: "Log in", href: "/login" }}
        />
      </AuthCard>
      <MobileWarningBanner />
      <EmailNotAllowedModal
        isOpen={showNotAllowedModal}
        onClose={handleCloseNotAllowedModal}
      />
    </div>
  );
}
