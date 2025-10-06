"use client";

import { BehaveAs, getBehaveAs } from "@/lib/utils";
import { Turnstile, TurnstileInstance } from "@marsidev/react-turnstile";
import { useEffect, useRef } from "react";

const TURNSTILE_SITE_KEY =
  process.env.NEXT_PUBLIC_CLOUDFLARE_TURNSTILE_SITE_KEY || "";

type Props = {
  onVerified: (token: string) => void;
  onReady: (ref: TurnstileInstance) => void;
  visible: boolean;
};

export function Turnstile2(props: Props) {
  const captchaRef = useRef<TurnstileInstance>(null);
  const behaveAs = getBehaveAs();

  useEffect(() => {
    if (captchaRef.current) {
      props.onReady(captchaRef.current);
    }
  }, [captchaRef]);

  function handleCaptchaVerify(token: string) {
    props.onVerified(token);
  }

  if (behaveAs !== BehaveAs.CLOUD) {
    return null;
  }

  if (!TURNSTILE_SITE_KEY) {
    return null;
  }

  return (
    <div className={props.visible ? "" : "hidden"}>
      <Turnstile
        ref={captchaRef}
        siteKey={TURNSTILE_SITE_KEY}
        onSuccess={handleCaptchaVerify}
      />
    </div>
  );
}
