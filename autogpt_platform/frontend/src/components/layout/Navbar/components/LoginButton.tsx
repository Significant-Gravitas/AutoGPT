"use client";

import { Button } from "@/components/atoms/Button/Button";
import { usePathname, useRouter } from "next/navigation";
import { Login03Icon } from "@hugeicons/core-free-icons";

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
      size="md"
      leadingIcon={Login03Icon}
      variant="secondary"
    >
      Log In
    </Button>
  );
}
