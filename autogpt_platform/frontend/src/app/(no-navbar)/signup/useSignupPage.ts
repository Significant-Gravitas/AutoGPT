import { useToast } from "@/components/molecules/Toast/use-toast";
import { useCaptureMarketingPrompt } from "@/hooks/useCaptureMarketingPrompt";
import { sanitizeAuthNext } from "@/lib/auth-redirect";
import { useAuth } from "@/lib/auth/hooks/useAuth";
import { setMarketingOptOutFlag } from "@/services/analytics/marketing-opt-out-cookie";
import { trackSignupMarketingOptOut } from "@/services/analytics/signup-analytics";
import { environment } from "@/services/environment";
import { LoginProvider, signupFormSchema } from "@/types/auth";
import { zodResolver } from "@hookform/resolvers/zod";
import { useRouter, useSearchParams } from "next/navigation";
import { useEffect, useState } from "react";
import { useForm } from "react-hook-form";
import z from "zod";
import { signup as signupAction } from "./actions";

export function useSignupPage() {
  useCaptureMarketingPrompt();

  const { user, isUserLoading, isLoggedIn } = useAuth();
  const [hasInitializedAuth, setHasInitializedAuth] = useState(false);
  const [feedback, setFeedback] = useState<string | null>(null);
  const { toast } = useToast();
  const router = useRouter();
  const searchParams = useSearchParams();
  const [isLoading, setIsLoading] = useState(false);
  const [isSigningUp, setIsSigningUp] = useState(false);
  const [isGoogleLoading, setIsGoogleLoading] = useState(false);
  const [showNotAllowedModal, setShowNotAllowedModal] = useState(false);
  const [verificationEmail, setVerificationEmail] = useState<string | null>(
    null,
  );
  const isCloudEnv = environment.isCloud();

  // Same-origin redirect target; off-site values are dropped so a crafted
  // `/signup?next=https://phishing.site` cannot redirect users elsewhere.
  const nextUrl = sanitizeAuthNext(searchParams.get("next"));

  useEffect(() => {
    if (!isUserLoading) {
      setHasInitializedAuth(true);
    }
  }, [isUserLoading]);

  // Only honour explicit `?next=` deep links here. Generic "already logged in
  // on /signup, get me out" is handled by OnboardingProvider so the user lands
  // straight on /onboarding or /copilot. Otherwise we'd bounce
  // /signup → / → /copilot → /onboarding, and each hop renders before the next
  // redirect — that intermediate /copilot render is the flash users see.
  useEffect(() => {
    if (isLoggedIn && !isSigningUp && nextUrl) {
      router.replace(nextUrl);
    }
  }, [isLoggedIn, isSigningUp, nextUrl, router]);

  const form = useForm<
    z.input<typeof signupFormSchema>,
    unknown,
    z.output<typeof signupFormSchema>
  >({
    resolver: zodResolver(signupFormSchema),
    defaultValues: {
      email: "",
      password: "",
      confirmPassword: "",
      marketingOptOut: false,
    },
  });

  function handleToggleMarketingOptOut() {
    if (isSigningUp) return;
    const optOut = !form.getValues("marketingOptOut");
    form.setValue("marketingOptOut", optOut, { shouldDirty: true });
    if (optOut) trackSignupMarketingOptOut();
  }

  async function handleProviderSignup(provider: LoginProvider) {
    setIsGoogleLoading(true);
    setIsSigningUp(true);

    try {
      // Include next URL in OAuth flow if present
      const callbackUrl = nextUrl
        ? `/auth/callback?next=${encodeURIComponent(nextUrl)}`
        : `/auth/callback`;
      const fullCallbackUrl = `${window.location.origin}${callbackUrl}`;

      setMarketingOptOutFlag(form.getValues("marketingOptOut") ?? false);

      const response = await fetch("/api/auth/login/with-provider", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ provider, redirectTo: fullCallbackUrl }),
      });

      if (!response.ok) {
        const { error } = await response.json();

        if (error === "not_allowed") {
          setShowNotAllowedModal(true);
          setIsSigningUp(false);
          return;
        }

        throw new Error(error || "Failed to start OAuth flow");
      }

      const { url } = await response.json();
      if (url) window.location.href = url as string;
    } catch (error) {
      setIsGoogleLoading(false);
      setIsSigningUp(false);
      toast({
        title:
          error instanceof Error ? error.message : "Failed to start OAuth flow",
        variant: "destructive",
      });
    }
  }

  async function handleSignup(data: z.output<typeof signupFormSchema>) {
    // The server action records this signup's choice; the cookie is only for
    // the Google round trip. Clear it so neither this choice nor one left from
    // an abandoned Google attempt is applied to a later Google sign-in.
    setMarketingOptOutFlag(false);
    setIsLoading(true);

    if (data.email.includes("@agpt.co")) {
      toast({
        title:
          "Please use Google SSO to create an account using an AutoGPT email.",
        variant: "default",
      });

      setIsLoading(false);
      return;
    }

    setIsSigningUp(true);

    try {
      const result = await signupAction(
        data.email,
        data.password,
        data.confirmPassword,
        data.marketingOptOut,
        nextUrl,
      );

      if (!result.success) {
        if (result.error === "user_already_exists") {
          setFeedback("User with this email already exists");
          setIsSigningUp(false);
          return;
        }
        if (result.error === "not_allowed") {
          setShowNotAllowedModal(true);
          setIsSigningUp(false);
          return;
        }

        toast({
          title: result.error || "Signup failed",
          variant: "destructive",
        });
        setIsSigningUp(false);
        return;
      }

      if (result.verificationRequired) {
        // There is no session yet, so the action recorded nothing. The emailed
        // link carries the refusal and lands on /auth/callback, which records
        // it with the terms.
        setVerificationEmail(result.email);
        setIsLoading(false);
        setIsSigningUp(false);
        return;
      }

      // Prefer the URL's next parameter, then result.next (for onboarding), then default
      const redirectTo = nextUrl || result.next || "/";
      router.replace(redirectTo);
    } catch (error) {
      setIsLoading(false);
      setIsSigningUp(false);
      toast({
        title:
          error instanceof Error
            ? error.message
            : "Unexpected error during signup",
        variant: "destructive",
      });
    } finally {
      setTimeout(() => {
        setIsLoading(false);
      }, 3000);
    }
  }

  function handleStartAgain() {
    setVerificationEmail(null);
    form.resetField("email");
  }

  return {
    form,
    feedback,
    nextUrl,
    verificationEmail,
    isLoggedIn: !!user,
    hasInitializedAuth,
    isLoading,
    isGoogleLoading,
    isSigningUp,
    isCloudEnv,
    isUserLoading,
    showNotAllowedModal,
    optedOut: form.watch("marketingOptOut") ?? false,
    handleSubmit: form.handleSubmit(handleSignup),
    handleToggleMarketingOptOut,
    handleCloseNotAllowedModal: () => setShowNotAllowedModal(false),
    handleProviderSignup,
    handleStartAgain,
  };
}
