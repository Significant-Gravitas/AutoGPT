import { useToast } from "@/components/molecules/Toast/use-toast";
import { useEffect, useState } from "react";
import {
  resendSignInCode,
  SIGN_IN_CODE_COPY,
  type SignInCodeEnd,
} from "../../helpers";

interface Args {
  email: string | null;
  resendAt: number;
  onEnded: (end: SignInCodeEnd) => void;
  onResent: () => void;
}

export function useResendCode({ email, resendAt, onEnded, onResent }: Args) {
  const { toast } = useToast();
  const [deadline, setDeadline] = useState(resendAt);
  const [now, setNow] = useState(() => Date.now());
  const [isResending, setIsResending] = useState(false);

  // Ticks from the clock, not by counting, so time a sleeping tab missed
  // still counts.
  useEffect(() => {
    if (now >= deadline) return;
    const timer = setTimeout(() => setNow(Date.now()), 1000);
    return () => clearTimeout(timer);
  }, [now, deadline]);

  const cooldown = Math.max(0, Math.ceil((deadline - now) / 1000));

  async function handleResend() {
    setIsResending(true);
    const outcome = await resendSignInCode().catch(() => null);
    setIsResending(false);
    if (outcome?.kind === "sent") {
      setNow(Date.now());
      setDeadline(Date.now() + outcome.resendInSeconds * 1000);
      onResent();
      toast({
        title: email ? `New code sent to ${email}` : "New code sent",
        variant: "success",
      });
    } else if (outcome?.kind === "ended") {
      onEnded(outcome.end);
    } else {
      const title = outcome?.message ?? SIGN_IN_CODE_COPY.failed;
      toast({ title, variant: "destructive" });
    }
  }

  return {
    cooldown,
    isResending,
    canResend: cooldown <= 0 && !isResending,
    handleResend,
  };
}
