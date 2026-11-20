"use client";

import { useState } from "react";
import { Button } from "@/components/atoms/Button/Button";
import { toast } from "@/components/molecules/Toast/use-toast";
import { useAuth } from "@/lib/auth/hooks/useAuth";
import { Copy01Icon, Tick02Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

// Renders the right-slot CTAs on every ``/share/...`` viewer:
//
//   - **Copy link** — always present.  Anonymous viewers want a one-
//     click handoff (paste into a chat / email); signed-in viewers
//     get the same convenience.
//   - **Sign up** — only when the viewer is not authenticated.
//     Hidden for signed-in users so the header doesn't nag them to
//     re-sign-up.
//
// Lives next to ``ShareHeader`` because both share routes (execution
// and chat) plug into the same header.  Keeping the CTAs in one
// component means a future addition lands in one place.
export function ShareActions() {
  const { isLoggedIn, isUserLoading } = useAuth();
  const [copied, setCopied] = useState(false);

  async function handleCopy() {
    if (typeof window === "undefined") return;
    try {
      await navigator.clipboard.writeText(window.location.href);
      setCopied(true);
      // 2s matches the share dialog's copy-flash so the two flows
      // feel consistent.
      setTimeout(() => setCopied(false), 2000);
    } catch {
      toast({
        title: "Failed to copy link",
        variant: "destructive",
      });
    }
  }

  return (
    <>
      <Button
        size="small"
        variant="secondary"
        onClick={handleCopy}
        leftIcon={
          copied ? (
            <Icon icon={Tick02Icon} size={14} />
          ) : (
            <Icon icon={Copy01Icon} size={14} />
          )
        }
      >
        {copied ? "Copied" : "Copy link"}
      </Button>
      {/* While the auth state is still loading we suppress the Sign up
          CTA — flickering it in then out on hydration looks broken to
          signed-in users opening their own share. */}
      {!isUserLoading && !isLoggedIn && (
        <Button size="small" variant="primary" as="NextLink" href="/signup">
          Sign up
        </Button>
      )}
    </>
  );
}
