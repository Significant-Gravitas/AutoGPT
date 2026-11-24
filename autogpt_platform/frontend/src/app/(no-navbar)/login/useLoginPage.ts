import { useHasSessionCookie } from "@/components/auth/SessionCookieProvider";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { useCaptureMarketingPrompt } from "@/hooks/useCaptureMarketingPrompt";
import { useServerRenderedForm } from "@/hooks/useServerRenderedForm";
import { getAuthDeepLink, sanitizeAuthNext } from "@/lib/auth-redirect";
import { providerCallbackPath } from "@/lib/auth/provider-callback";
import { isBannedNotice } from "@/lib/auth/account-banned";
import {
  EMAIL_VERIFICATION_NOTICE_PARAM,
  getEmailVerificationNotice,
} from "@/lib/auth/email-verification";
import { useAuth } from "@/lib/auth/hooks/useAuth";
import {
  getSocialSignInError,
  isWaitlistSignInError,
} from "@/lib/auth/microsoft-login";
import { environment } from "@/services/environment";
import { loginFormSchema, LoginProvider } from "@/types/auth";
import { zodResolver } from "@hookform/resolvers/zod";
import { useRouter, useSearchParams } from "next/navigation";
import { useEffect, useState } from "react";
import { useForm } from "react-hook-form";
import z from "zod";
import { login as loginAction } from "./actions";
import { getPostLoginPath, getSignInErrorNotice } from "./helpers";
import { isTeamEmail } from "@/lib/auth/team-email";
import { SIGN_IN_CODE_SENT } from "@/lib/auth/sign-in-code";
import { clearImpersonationEverywhere } from "@/lib/impersonation";
import { rememberConnectAttempt } from "@/lib/auth/connect-attempt";
import { useConnectFallback } from "@/components/auth/useConnectFallback";
import { usePageRestored } from "@/components/auth/usePageRestored";

export function useLoginPage() {
  useCaptureMarketingPrompt();

  const { user, isUserLoading, isLoggedIn } = useAuth();
  const hasSessionCookie = useHasSessionCookie();
  const [feedback, setFeedback] = useState<string | null>(null);
  const router = useRouter();
  const searchParams = useSearchParams();
  const { toast } = useToast();
  const [isLoading, setIsLoading] = useState(false);
  const [isLoggingIn, setIsLoggingIn] = useState(false);
  const [loadingProvider, setLoadingProvider] = useState<LoginProvider | null>(
    null,
  );
  // Back from the provider can bring the page back as it was left, still
  // starting that sign-in.
  usePageRestored(() => {
    setLoadingProvider(null);
    setIsLoggingIn(false);
  });
  // A social sign-in that fails returns here with Better Auth's error code.
  // Microsoft's own copy (microsoft-login.ts) is for a Microsoft sign-in only,
  // which the with-provider route marks with `provider=microsoft`.
  const microsoftErrorCode =
    searchParams.get("provider") === "microsoft"
      ? searchParams.get("error")
      : null;
  const isWaitlisted = isWaitlistSignInError(microsoftErrorCode);
  const connectFallback = useConnectFallback();
  const [showNotAllowedModal, setShowNotAllowedModal] = useState(isWaitlisted);
  const [verificationEmail, setVerificationEmail] = useState<string | null>(
    null,
  );
  const isCloudEnv = environment.isCloud();

  // Same-origin redirect target; off-site values are dropped so a crafted
  // `/login?next=https://phishing.site` cannot redirect users elsewhere.
  const rawNext = searchParams.get("next");
  const nextUrl = getAuthDeepLink(rawNext);
  // Round trips (Google, the verification email) carry the whole safe `next`;
  // /auth/callback decides from it, tracking params included.
  const safeNext = sanitizeAuthNext(rawNext);
  const verificationNotice = getEmailVerificationNotice(
    searchParams.get(EMAIL_VERIFICATION_NOTICE_PARAM),
  );
  const signInErrorNotice = getSignInErrorNotice(
    searchParams.get("error"),
    searchParams.get("provider"),
  );
  // A provider sign-in that can't vouch for its address waits on a code
  // emailed there (require-sign-in-code.ts).
  const isWaitingOnCode = searchParams.get("error") === SIGN_IN_CODE_SENT;
  const showBannedNotice = isBannedNotice(searchParams);

  // Only honour explicit `?next=` deep links here. Generic "already logged in
  // on /login, get me out" is handled by OnboardingProvider so the user lands
  // straight on /onboarding or /copilot. Otherwise we'd bounce
  // /login → / → /copilot → /onboarding, and each hop renders before the next
  // redirect — that intermediate /copilot render is the flash users see.
  useEffect(() => {
    if (isLoggedIn && !isLoggingIn && nextUrl) {
      router.replace(nextUrl);
    }
  }, [isLoggedIn, isLoggingIn, nextUrl, router]);

  const form = useForm<z.infer<typeof loginFormSchema>>({
    resolver: zodResolver(loginFormSchema),
    defaultValues: {
      email: "",
      password: "",
    },
  });
  const { formRef, isHydrated } = useServerRenderedForm(form);

  async function handleProviderLogin(
    provider: LoginProvider,
    { connect }: { connect?: boolean } = {},
  ) {
    setLoadingProvider(provider);
    setIsLoggingIn(true);
    // Leftover impersonation (expired admin session, pre-fix cookie) would 403 a non-admin.
    clearImpersonationEverywhere();

    try {
      // Include next URL in OAuth flow if present
      const fullCallbackUrl = `${window.location.origin}${providerCallbackPath(provider, safeNext)}`;

      const response = await fetch("/api/auth/login/with-provider", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          provider,
          redirectTo: fullCallbackUrl,
          connect,
        }),
      });

      if (!response.ok) {
        const { error } = await response.json();
        throw new Error(error || "Failed to start OAuth flow");
      }

      const { url, connecting } = await response.json();
      if (connecting) rememberConnectAttempt(provider);
      if (url) window.location.href = url as string;
    } catch (error) {
      setLoadingProvider(null);
      setIsLoggingIn(false);
      setFeedback(
        error instanceof Error ? error.message : "Failed to start OAuth flow",
      );
    }
  }

  async function handleLogin(data: z.infer<typeof loginFormSchema>) {
    setIsLoading(true);
    setIsLoggingIn(true);

    if (isTeamEmail(data.email)) {
      toast({
        title: "Please use Google SSO to login using an AutoGPT email.",
        variant: "default",
      });

      setIsLoading(false);
      setIsLoggingIn(false);
      return;
    }

    clearImpersonationEverywhere();

    try {
      const result = await loginAction(data.email, data.password, safeNext);

      if (result.error === "email_not_verified" && result.email) {
        setVerificationEmail(result.email);
        setIsLoading(false);
        setIsLoggingIn(false);
        return;
      }

      if (!result.success) {
        throw new Error(result.error || "Login failed");
      }

      // Use full page navigation to ensure middleware processes the new auth cookies.
      // router.replace() does a soft navigation where the cookie store may not
      // immediately reflect cookies set by the server action, causing a blank page.
      // This matches the OAuth flow which also uses window.location.href.
      window.location.href = getPostLoginPath({
        nextUrl,
        resultNext: result.next,
        rawNext,
      });
    } catch (error) {
      toast({
        title:
          error instanceof Error
            ? error.message
            : "Unexpected error during login",
        variant: "destructive",
      });

      setIsLoading(false);
      setIsLoggingIn(false);
    }
  }

  // Back to the form, keeping only the deep link.
  function handleLeaveSignInCode() {
    router.replace(
      safeNext ? `/login?next=${encodeURIComponent(safeNext)}` : "/login",
    );
  }

  return {
    form,
    formRef,
    feedback:
      feedback ??
      signInErrorNotice ??
      (isWaitlisted || isWaitingOnCode
        ? null
        : getSocialSignInError(microsoftErrorCode)),
    nextUrl: safeNext,
    verificationEmail,
    verificationNotice,
    isWaitingOnCode,
    showBannedNotice,
    user,
    isLoading,
    isGoogleLoading: loadingProvider === "google",
    isMicrosoftLoading: loadingProvider === "microsoft",
    isMicrosoftEnabled: environment.isMicrosoftLoginEnabled(),
    isCloudEnv,
    // Without a session cookie the visitor is signed out, so the form shows at
    // once; with one, it waits for the session check.
    isCheckingSession: hasSessionCookie && isUserLoading,
    isHydrated,
    showNotAllowedModal,
    handleSubmit: form.handleSubmit(handleLogin),
    handleProviderLogin,
    connectFallback,
    handleCloseNotAllowedModal: () => setShowNotAllowedModal(false),
    handleBackToLogin: () => setVerificationEmail(null),
    handleLeaveSignInCode,
  };
}
