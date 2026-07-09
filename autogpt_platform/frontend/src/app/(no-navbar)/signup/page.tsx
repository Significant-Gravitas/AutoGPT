"use client";

import { Checkbox } from "@/components/__legacy__/ui/checkbox";
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
} from "@/components/__legacy__/ui/form";
import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { Link } from "@/components/atoms/Link/Link";
import { Text } from "@/components/atoms/Text/Text";
import AuthFeedback from "@/components/auth/AuthFeedback";
import { EmailNotAllowedModal } from "@/components/auth/EmailNotAllowedModal";
import { GoogleOAuthButton } from "@/components/auth/GoogleOAuthButton";
import { AuthDivider } from "@/components/auth/AuthSplitLayout/AuthDivider";
import { AuthSplitLayout } from "@/components/auth/AuthSplitLayout/AuthSplitLayout";
import { MobileWarningBanner } from "@/components/auth/MobileWarningBanner";
import { environment } from "@/services/environment";
import { WarningOctagonIcon } from "@phosphor-icons/react/dist/ssr";
import { useSearchParams } from "next/navigation";
import { LoadingSignup } from "./components/LoadingSignup";
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
    isLoggedIn,
    isLoading,
    isGoogleLoading,
    isCloudEnv,
    isUserLoading,
    showNotAllowedModal,
    isSupabaseAvailable,
    handleSubmit,
    handleProviderSignup,
    handleCloseNotAllowedModal,
  } = useSignupPage();

  if (isUserLoading || isLoggedIn) {
    return <LoadingSignup />;
  }

  if (!isSupabaseAvailable) {
    return (
      <AuthSplitLayout marketing={<SignupMarketingPanel />}>
        <Text variant="body-medium" className="text-center !text-slate-500">
          User accounts are disabled because Supabase client is unavailable
        </Text>
      </AuthSplitLayout>
    );
  }

  const confirmPasswordError = form.formState.errors.confirmPassword?.message;
  const termsError = form.formState.errors.agreeToTerms?.message;

  return (
    <AuthSplitLayout marketing={<SignupMarketingPanel />}>
      <div className="mb-8">
        <Text variant="h3" as="h1" className="!text-slate-950">
          Create your account
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
          <FormField
            control={form.control}
            name="agreeToTerms"
            render={({ field }) => (
              <>
                <FormItem className="mt-6 flex w-full flex-row items-center -space-y-1 space-x-2">
                  <FormControl>
                    <Checkbox
                      checked={field.value}
                      onCheckedChange={field.onChange}
                      className="relative bottom-px"
                    />
                  </FormControl>
                  <div>
                    <FormLabel className="flex flex-wrap items-center gap-1">
                      <Text
                        variant="body-medium"
                        className="inline-block text-slate-950"
                      >
                        I agree to the
                      </Text>
                      <Link
                        href="https://agpt.co/legal/platform-terms-of-use"
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
                        href="https://agpt.co/legal/platform-privacy-policy"
                        variant="secondary"
                      >
                        Privacy Policy
                      </Link>
                    </FormLabel>
                  </div>
                </FormItem>
                {termsError ? (
                  <div className="flex items-center gap-2">
                    <WarningOctagonIcon className="h-4 w-4 text-red-500" />
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

        <AuthFeedback
          type="signup"
          message={feedback}
          isError={!!feedback}
          behaveAs={environment.getBehaveAs()}
        />
      </Form>

      <div className="mt-6 inline-flex w-full items-center justify-center gap-1">
        <Text variant="body-medium" className="!text-slate-500">
          Already a member?
        </Text>
        <Link href={loginHref} variant="secondary">
          Log in
        </Link>
      </div>

      <MobileWarningBanner />
      <EmailNotAllowedModal
        isOpen={showNotAllowedModal}
        onClose={handleCloseNotAllowedModal}
      />
    </AuthSplitLayout>
  );
}
