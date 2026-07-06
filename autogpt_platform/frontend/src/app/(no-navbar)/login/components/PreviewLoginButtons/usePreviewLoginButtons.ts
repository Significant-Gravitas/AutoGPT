import { useToast } from "@/components/molecules/Toast/use-toast";
import { sanitizeAuthNext } from "@/lib/auth-redirect";
import { environment } from "@/services/environment";
import { useSearchParams } from "next/navigation";
import { useEffect, useState } from "react";
import { isPreviewLoginConfigured, loginAsPreviewAccount } from "./actions";
import { PreviewRole } from "./helpers";

export function usePreviewLoginButtons() {
  const isPreview = Boolean(environment.getPreviewStealingDev());
  const [isConfigured, setIsConfigured] = useState(false);
  const [isCheckingConfig, setIsCheckingConfig] = useState(true);
  const [loadingRole, setLoadingRole] = useState<PreviewRole | null>(null);
  const { toast } = useToast();
  const searchParams = useSearchParams();
  const nextUrl = sanitizeAuthNext(searchParams.get("next"));

  useEffect(() => {
    if (!isPreview) return;

    let active = true;
    isPreviewLoginConfigured().then((configured) => {
      if (!active) return;
      setIsConfigured(configured);
      setIsCheckingConfig(false);
    });

    return () => {
      active = false;
    };
  }, [isPreview]);

  async function handlePreviewLogin(role: PreviewRole) {
    setLoadingRole(role);

    try {
      const result = await loginAsPreviewAccount(role);
      if (!result.success) {
        throw new Error(result.error || "Preview login failed");
      }

      // Full page navigation so middleware picks up the new auth cookies,
      // mirroring the standard email/password login flow.
      window.location.href = nextUrl || result.next || "/";
    } catch (error) {
      toast({
        title: error instanceof Error ? error.message : "Preview login failed",
        variant: "destructive",
      });
      setLoadingRole(null);
    }
  }

  return {
    isPreview,
    isConfigured,
    isCheckingConfig,
    loadingRole,
    handlePreviewLogin,
  };
}
