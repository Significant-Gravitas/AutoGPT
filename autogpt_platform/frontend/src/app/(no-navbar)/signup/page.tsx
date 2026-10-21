"use client";

import { Form, FormField } from "@/components/__legacy__/ui/form";
import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { Link } from "@/components/atoms/Link/Link";
import { Text } from "@/components/atoms/Text/Text";
import AuthFeedback from "@/components/auth/AuthFeedback";
import { CheckYourInbox } from "@/components/auth/CheckYourInbox/CheckYourInbox";
import { EmailNotAllowedModal } from "@/components/auth/EmailNotAllowedModal";
import { GoogleOAuthButton } from "@/components/auth/GoogleOAuthButton";
import { AuthDivider } from "@/components/auth/AuthSplitLayout/AuthDivider";
import { AuthSplitLayout } from "@/components/auth/AuthSplitLayout/AuthSplitLayout";
import { MobileWarningBanner } from "@/components/auth/MobileWarningBanner";
import { environment } from "@/services/environment";
import { useSearchParams } from "next/navigation";
import { LoadingSignup } from "./components/LoadingSignup";
import { SignupLegalLine } from "./components/SignupLegalLine";
import { SignupMarketingPanel } from "./components/SignupMarketingPanel";
import { useSignupPage } from "./useSignupPage";

export const dynamic = "force-dynamic";

export default function SignupPage() {
  const searchParams = useSearchParams();
  const nextUrl = searchParams.get("next");
  const loginHref = nextUrl
    ? `/login?next=${encodeURIComponent(nextUrl)}`
    : "/login";

  const {
    form,
    feedback,
    nextUrl: safeNextUrl,
    verificationEmail,
    isLoggedIn,
    hasInitializedAuth,
    isLoading,
    isGoogleLoading,
    isSigningUp,
    isCloudEnv,
    showNotAllowedModal,
    optedOut,
    handleSubmit,
    handleToggleMarketingOptOut,
    handleProviderSignup,
    handleCloseNotAllowedModal,
    handleStartAgain,
  } = useSignupPage();

  if (!hasInitializedAuth || isLoggedIn) {
    return <LoadingSignup />;
  }

  if (verificationEmail) {
    return (
      <AuthSplitLayout marketing={<SignupMarketingPanel />}>
        <CheckYourInbox
          email={verificationEmail}
          reason="signup"
          next={safeNextUrl}
          marketingOptOut={optedOut}
          onBack={handleStartAgain}
        />
      </AuthSplitLayout>
    );
  }

  const confirmPasswordError = form.formState.errors.confirmPassword?.message;

  return (
    <AuthSplitLayout marketing={<SignupMarketingPanel />}>
      <div className="mb-8">
        <Text variant="h3" as="h1" className="!text-slate-950">
          Create your account
        </Text>
        <Text variant="body" className="mt-1 !text-slate-500">
          Already a member?{" "}
          <Link href={loginHref} variant="secondary">
            Log in
          </Link>
        </Text>
      </div>

      <Form {...form}>
        <form onSubmit={handleSubmit} className="flex w-full flex-col gap-1">
          <FormField
            control={form.control}
            name="email"
            render={({ field }) => (
              <Input
                id={field.name}
                label="Email"
                placeholder="name@company.com"
                type="email"
                autoComplete="email"
                error={form.formState.errors.email?.message}
                {...field}
              />
            )}
          />
          <FormField
            control={form.control}
            name="password"
            render={({ field }) => (
              <Input
                id={field.name}
                label="Password"
                placeholder="Create a password"
                type="password"
                autoComplete="new-password"
                error={form.formState.errors.password?.message}
                {...field}
              />
            )}
          />
          <FormField
            control={form.control}
            name="confirmPassword"
            render={({ field }) => (
              <Input
                id={field.name}
                label="Confirm Password"
                placeholder="Confirm your password"
                type="password"
                autoComplete="new-password"
                error={confirmPasswordError}
                {...field}
              />
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

        {isCloudEnv ? (
          <>
            <AuthDivider />
            <GoogleOAuthButton
              onClick={() => handleProviderSignup("google")}
              isLoading={isGoogleLoading}
              disabled={isLoading}
            />
          </>
        ) : null}

        <SignupLegalLine
          optedOut={optedOut}
          onToggle={handleToggleMarketingOptOut}
          disabled={isSigningUp}
        />

        <AuthFeedback
          type="signup"
          message={feedback}
          isError={!!feedback}
          behaveAs={environment.getBehaveAs()}
        />
      </Form>

      <MobileWarningBanner />
      <EmailNotAllowedModal
        isOpen={showNotAllowedModal}
        onClose={handleCloseNotAllowedModal}
      />
    </AuthSplitLayout>
  );
}
