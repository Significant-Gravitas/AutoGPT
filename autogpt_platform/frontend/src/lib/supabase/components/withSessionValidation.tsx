"use client";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { useSupabase } from "../hooks/useSupabase";
import LoadingBox from "@/components/ui/loading";

interface WithSessionValidationOptions {
  redirectTo?: string;
  requireAuth?: boolean;
  adminOnly?: boolean;
}

export function withSessionValidation<P extends object>(
  WrappedComponent: React.ComponentType<P>,
  options: WithSessionValidationOptions = {},
) {
  const {
    redirectTo = "/login",
    requireAuth = true,
    adminOnly = false,
  } = options;

  return function SessionValidatedComponent(props: P) {
    const { user, isUserLoading, validateSession } = useSupabase();
    const [isValidating, setIsValidating] = useState(false);
    const router = useRouter();

    useEffect(() => {
      if (isUserLoading) return;

      async function performValidation() {
        setIsValidating(true);
        const isValid = await validateSession();

        if (requireAuth && (!isValid || !user)) {
          router.push(redirectTo);
          return;
        }

        if (adminOnly && user?.role !== "admin") {
          router.push("/marketplace");
          return;
        }

        setIsValidating(false);
      }

      performValidation();
    }, [user, isUserLoading]);

    if (isUserLoading || isValidating) {
      return <LoadingBox className="h-[80vh]" />;
    }

    if (requireAuth && !user) return null;
    if (adminOnly && user?.role !== "admin") return null;

    return <WrappedComponent {...props} />;
  };
}

export function withAdminValidation<P extends object>(
  WrappedComponent: React.ComponentType<P>,
) {
  return withSessionValidation(WrappedComponent, {
    requireAuth: true,
    adminOnly: true,
    redirectTo: "/marketplace",
  });
}

export function withAuthValidation<P extends object>(
  WrappedComponent: React.ComponentType<P>,
) {
  return withSessionValidation(WrappedComponent, {
    requireAuth: true,
  });
}
