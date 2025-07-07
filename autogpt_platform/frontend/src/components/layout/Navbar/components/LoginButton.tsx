"use client";

import { Button } from "@/components/atoms/Button/Button";
import { SignInIcon } from "@phosphor-icons/react/dist/ssr";
import { usePathname } from "next/navigation";

export function LoginButton() {
  const pathname = usePathname();
  const isLoginPage = pathname.includes("/login");

  if (isLoginPage) return null;

  return (
    <Button
      as="NextLink"
      href="/login"
      size="small"
      className="flex items-center justify-end space-x-2"
      leftIcon={<SignInIcon className="h-5 w-5" />}
      variant="secondary"
    >
      Log In
    </Button>
  );
}
