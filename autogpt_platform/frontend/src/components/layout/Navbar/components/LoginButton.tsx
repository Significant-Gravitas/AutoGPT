"use client";

import { Button } from "@/components/atoms/Button/Button";
import { SignInIcon } from "@phosphor-icons/react/dist/ssr";
import { usePathname, useRouter } from "next/navigation";

export function LoginButton() {
  const router = useRouter();
  const pathname = usePathname();
  const isLoginPage = pathname.includes("/login");

  if (isLoginPage) return null;

  function handleLogin() {
    router.push("/login");
  }

  return (
    <Button
      onClick={handleLogin}
      size="small"
      leftIcon={<SignInIcon className="size-4" />}
      variant="secondary"
    >
      Log In
    </Button>
  );
}
