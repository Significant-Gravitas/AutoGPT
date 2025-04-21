import { useState, useCallback, useEffect } from "react";
import { verifyTurnstileToken } from "@/lib/turnstile";
import { BehaveAs, getBehaveAs } from "@/lib/utils";

interface UseTurnstileOptions {
  action?: string;
  autoVerify?: boolean;
  onSuccess?: () => void;
  onError?: (error: Error) => void;
}

interface UseTurnstileResult {
  token: string | null;
  verifying: boolean;
  verified: boolean;
  error: Error | null;
  handleVerify: (token: string) => Promise<boolean>;
  handleExpire: () => void;
  handleError: (error: Error) => void;
  reset: () => void;
  siteKey: string;
  shouldRender: boolean;
}

const TURNSTILE_SITE_KEY = process.env.NEXT_PUBLIC_CLOUDFLARE_TURNSTILE_SITE_KEY || ''; 

/**
 * Custom hook for managing Turnstile state in forms
 */
export function useTurnstile({
  action,
  autoVerify = true,
  onSuccess,
  onError,
}: UseTurnstileOptions = {}): UseTurnstileResult {
  const [token, setToken] = useState<string | null>(null);
  const [verifying, setVerifying] = useState(false);
  const [verified, setVerified] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [shouldRender, setShouldRender] = useState(false);
  
  useEffect(() => {
    const behaveAs = getBehaveAs();
    const hasTurnstileKey = !!TURNSTILE_SITE_KEY;

    setShouldRender(behaveAs === BehaveAs.CLOUD && hasTurnstileKey);
    
    if (behaveAs !== BehaveAs.CLOUD || !hasTurnstileKey) {
      setVerified(true);
    }
  }, []);
  
  useEffect(() => {
    if (token && !autoVerify && shouldRender) {
      setVerified(true);
    }
  }, [token, autoVerify, shouldRender]);
  
  const reset = useCallback(() => {
    if (shouldRender) {
      setToken(null);
      setVerifying(false);
      setVerified(false);
      setError(null);
    }
  }, [shouldRender]);
  
  const handleVerify = useCallback(
    async (newToken: string) => {
      if (!shouldRender) {
        return true;
      }
      
      setToken(newToken);
      setError(null);
      
      if (autoVerify) {
        setVerifying(true);
        
        try {
          const success = await verifyTurnstileToken(newToken, action);
          setVerified(success);
          
          if (success && onSuccess) {
            onSuccess();
          } else if (!success) {
            const newError = new Error("Turnstile verification failed");
            setError(newError);
            if (onError) onError(newError);
          }
          
          setVerifying(false);
          return success;
        } catch (err) {
          const newError = err instanceof Error ? err : new Error("Unknown error during verification");
          setError(newError);
          setVerified(false);
          setVerifying(false);
          if (onError) onError(newError);
          return false;
        }
      } else {
        setVerified(true);
      }
      
      return true;
    },
    [action, autoVerify, onSuccess, onError, shouldRender]
  );
  
  const handleExpire = useCallback(() => {
    if (shouldRender) {
      setToken(null);
      setVerified(false);
    }
  }, [shouldRender]);
  
  const handleError = useCallback(
    (err: Error) => {
      if (shouldRender) {
        setError(err);
        setVerified(false);
        if (onError) onError(err);
      }
    },
    [onError, shouldRender]
  );
  
  return {
    token,
    verifying,
    verified,
    error,
    handleVerify,
    handleExpire,
    handleError,
    reset,
    siteKey: TURNSTILE_SITE_KEY,
    shouldRender,
  };
} 