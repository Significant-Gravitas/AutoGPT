import React from "react";
import { Text } from "@/components/atoms/Text/Text";
import { SignIn, UserCircle } from "@phosphor-icons/react";
import { Button } from "@/components/atoms/Button/Button";
import { cn } from "@/lib/utils";

export interface LoginPromptProps {
  message: string;
  onLogin: () => void;
  onContinueAsGuest: () => void;
  className?: string;
}

export function LoginPrompt({
  message,
  onLogin,
  onContinueAsGuest,
  className,
}: LoginPromptProps) {
  return (
    <div
      className={cn(
        "mx-4 my-2 flex flex-col items-center gap-4 rounded-lg border border-blue-200 bg-blue-50 p-6 dark:border-blue-900 dark:bg-blue-950",
        className,
      )}
    >
      {/* Icon */}
      <div className="flex h-12 w-12 items-center justify-center rounded-full bg-blue-500">
        <UserCircle size={32} weight="fill" className="text-white" />
      </div>

      {/* Content */}
      <div className="text-center">
        <Text variant="h3" className="mb-2 text-blue-900 dark:text-blue-100">
          Login Required
        </Text>
        <Text variant="body" className="text-blue-700 dark:text-blue-300">
          {message}
        </Text>
      </div>

      {/* Action Buttons */}
      <div className="flex gap-3">
        <Button
          onClick={onLogin}
          variant="primary"
          className="flex items-center gap-2"
        >
          <SignIn size={20} weight="bold" />
          Login
        </Button>
        <Button onClick={onContinueAsGuest} variant="secondary">
          Continue as Guest
        </Button>
      </div>

      <Text variant="small" className="text-blue-600 dark:text-blue-400">
        Logging in will save your chat history to your account
      </Text>
    </div>
  );
}
