import { useToast } from "@/components/molecules/Toast/use-toast";
import { authClient } from "@/lib/auth/client";
import { getEmailVerificationCallbackURL } from "@/lib/auth/email-verification";
import { useEffect, useState } from "react";
import {
  CHECK_YOUR_INBOX_HEADING_ID,
  RESEND_COOLDOWN_SECONDS,
} from "./helpers";

interface Args {
  email: string;
  next?: string | null;
}

export function useCheckYourInbox({ email, next }: Args) {
  const { toast } = useToast();
  const [cooldown, setCooldown] = useState(RESEND_COOLDOWN_SECONDS);
  const [isResending, setIsResending] = useState(false);

  useEffect(() => {
    document.getElementById(CHECK_YOUR_INBOX_HEADING_ID)?.focus();
  }, []);

  useEffect(() => {
    if (cooldown <= 0) return;
    const timer = setTimeout(() => setCooldown((value) => value - 1), 1000);
    return () => clearTimeout(timer);
  }, [cooldown]);

  async function handleResend() {
    setIsResending(true);
    // Through the HTTP route rather than a server action so Better Auth's
    // per-IP rate limit applies.
    const { error } = await authClient.sendVerificationEmail({
      email,
      callbackURL: getEmailVerificationCallbackURL(next),
    });
    setIsResending(false);

    if (error) {
      toast({
        title: "We couldn't send the email",
        description:
          error.status === 429
            ? "Too many attempts. Please wait a minute and try again."
            : "Please try again in a moment.",
        variant: "destructive",
      });
      return;
    }

    setCooldown(RESEND_COOLDOWN_SECONDS);
    toast({ title: `Verification email sent to ${email}`, variant: "success" });
  }

  return {
    cooldown,
    isResending,
    canResend: cooldown <= 0 && !isResending,
    handleResend,
  };
}
