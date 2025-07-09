"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { Link } from "@/components/atoms/Link/Link";
import { Text } from "@/components/atoms/Text/Text";
import { AuthCard } from "@/components/auth/AuthCard";
import AuthFeedback from "@/components/auth/AuthFeedback";
import { EmailNotAllowedModal } from "@/components/auth/EmailNotAllowedModal";
import { GoogleOAuthButton } from "@/components/auth/GoogleOAuthButton";
import Turnstile from "@/components/auth/Turnstile";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
} from "@/components/ui/form";
import { getBehaveAs } from "@/lib/utils";
import { WarningOctagonIcon } from "@phosphor-icons/react/dist/ssr";
import { LoadingSignup } from "./components/LoadingSignup";
import { useSignupPage } from "./useSignupPage";

export default function SignupPage() {
  const {
    form,
    feedback,
    turnstile,
    captchaKey,
    isLoggedIn,
    isLoading,
    isCloudEnv,
    isUserLoading,
    isGoogleLoading,
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
      <div>
        User accounts are disabled because Supabase client is unavailable
      </div>
    );
  }

  const confirmPasswordError = form.formState.errors.confirmPassword?.message;
  const termsError = form.formState.errors.agreeToTerms?.message;

  return (
    <div className="mt-24 flex h-full min-h-[85vh] flex-col items-center justify-center md:mt-0">
      <AuthCard title="Create a new account">
        {isCloudEnv ? (
          <>
            <div className="mb-2 w-full">
              <GoogleOAuthButton
                onClick={() => handleProviderSignup("google")}
                isLoading={isGoogleLoading}
                disabled={isLoading}
              />
            </div>
            <div className="mb-3 flex w-full items-center">
              <div className="h-px flex-1 bg-gray-300"></div>
              <span className="text-md mx-3 text-gray-500">or</span>
              <div className="h-px flex-1 bg-gray-300"></div>
            </div>
          </>
        ) : null}

        <Form {...form}>
          <form onSubmit={handleSubmit} className="flex w-full flex-col gap-1">
            <FormField
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
            <FormField
              control={form.control}
              name="password"
              render={({ field }) => {
                console.log(field);
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
            <FormField
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

            {/* Turnstile CAPTCHA Component */}
            <Turnstile
              key={captchaKey}
              siteKey={turnstile.siteKey}
              onVerify={turnstile.handleVerify}
              onExpire={turnstile.handleExpire}
              onError={turnstile.handleError}
              setWidgetId={turnstile.setWidgetId}
              action="signup"
              shouldRender={turnstile.shouldRender}
            />

            <Button
              variant="primary"
              loading={isLoading}
              type="submit"
              className="mt-6 w-full"
            >
              {isLoading ? "Signing up..." : "Sign up"}
            </Button>
          </form>
        </Form>
        <AuthFeedback
          type="signup"
          message={feedback}
          isError={!!feedback}
          behaveAs={getBehaveAs()}
        />

        <AuthCard.BottomText
          text="Already a member?"
          link={{ text: "Log in", href: "/login" }}
        />
      </AuthCard>
      <EmailNotAllowedModal
        isOpen={showNotAllowedModal}
        onClose={handleCloseNotAllowedModal}
      />
    </div>
  );
}
