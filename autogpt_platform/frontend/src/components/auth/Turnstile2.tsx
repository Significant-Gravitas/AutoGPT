"use client";

import { BehaveAs, getBehaveAs } from "@/lib/utils";
import { Turnstile, TurnstileInstance } from "@marsidev/react-turnstile";
import { useEffect, useRef, useState } from "react";

const TURNSTILE_SITE_KEY =
  process.env.NEXT_PUBLIC_CLOUDFLARE_TURNSTILE_SITE_KEY || "";

type Props = {
  onVerified: (token: string) => void;
  onReady: (ref: TurnstileInstance) => void;
};

export function Turnstile2(props: Props) {
  const captchaRef = useRef<TurnstileInstance>(null);
  const [captchaToken, setCaptchaToken] = useState<string | null>(null);
  const behaveAs = getBehaveAs();

  useEffect(() => {
    if (captchaRef.current) {
      props.onReady(captchaRef.current);
    }
  }, [captchaRef]);

  function handleCaptchaVerify(token: string) {
    setCaptchaToken(token);
    props.onVerified(token);
  }

  // Only render in cloud environment
  if (behaveAs !== BehaveAs.CLOUD) {
    return null;
  }

  if (!TURNSTILE_SITE_KEY) {
    return null;
  }

  // If it is already verified, no need to render
  if (captchaToken) {
    return null;
  }

  return (
    <Turnstile
      ref={captchaRef}
      siteKey={TURNSTILE_SITE_KEY}
      onSuccess={handleCaptchaVerify}
    />
  );
}
