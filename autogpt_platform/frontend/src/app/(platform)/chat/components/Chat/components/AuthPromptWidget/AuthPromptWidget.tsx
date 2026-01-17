"use client";

import { Button } from "@/components/atoms/Button/Button";
import { cn } from "@/lib/utils";
import { ShieldIcon, SignInIcon, UserPlusIcon } from "@phosphor-icons/react";
import { useRouter } from "next/navigation";

export interface AuthPromptWidgetProps {
  message: string;
  sessionId: string;
  agentInfo?: {
    graph_id: string;
    name: string;
    trigger_type: string;
  };
  returnUrl?: string;
  className?: string;
}

export function AuthPromptWidget({
  message,
  sessionId,
  agentInfo,
  returnUrl = "/copilot/chat",
  className,
}: AuthPromptWidgetProps) {
  const router = useRouter();

  function handleSignIn() {
    if (typeof window !== "undefined") {
      localStorage.setItem("pending_chat_session", sessionId);
      if (agentInfo) {
        localStorage.setItem("pending_agent_setup", JSON.stringify(agentInfo));
      }
    }
    const returnUrlWithSession = `${returnUrl}?session_id=${sessionId}`;
    const encodedReturnUrl = encodeURIComponent(returnUrlWithSession);
    router.push(`/login?returnUrl=${encodedReturnUrl}`);
  }

  function handleSignUp() {
    if (typeof window !== "undefined") {
      localStorage.setItem("pending_chat_session", sessionId);
      if (agentInfo) {
        localStorage.setItem("pending_agent_setup", JSON.stringify(agentInfo));
      }
    }
    const returnUrlWithSession = `${returnUrl}?session_id=${sessionId}`;
    const encodedReturnUrl = encodeURIComponent(returnUrlWithSession);
    router.push(`/signup?returnUrl=${encodedReturnUrl}`);
  }

  return (
    <div
      className={cn(
        "my-4 overflow-hidden rounded-lg border border-violet-200",
        "bg-gradient-to-br from-violet-50 to-purple-50",
        "duration-500 animate-in fade-in-50 slide-in-from-bottom-2",
        className,
      )}
    >
      <div className="px-6 py-5">
        <div className="mb-4 flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-full bg-violet-600">
            <ShieldIcon size={20} weight="fill" className="text-white" />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-neutral-900">
              Authentication Required
            </h3>
            <p className="text-sm text-neutral-600">
              Sign in to set up and manage agents
            </p>
          </div>
        </div>

        <div className="mb-5 rounded-md bg-white/50 p-4">
          <p className="text-sm text-neutral-700">{message}</p>
          {agentInfo && (
            <div className="mt-3 text-xs text-neutral-600">
              <p>
                Ready to set up:{" "}
                <span className="font-medium">{agentInfo.name}</span>
              </p>
              <p>
                Type:{" "}
                <span className="font-medium">{agentInfo.trigger_type}</span>
              </p>
            </div>
          )}
        </div>

        <div className="flex gap-3">
          <Button
            onClick={handleSignIn}
            variant="primary"
            size="small"
            className="flex-1"
          >
            <SignInIcon size={16} weight="bold" className="mr-2" />
            Sign In
          </Button>
          <Button
            onClick={handleSignUp}
            variant="secondary"
            size="small"
            className="flex-1"
          >
            <UserPlusIcon size={16} weight="bold" className="mr-2" />
            Create Account
          </Button>
        </div>

        <div className="mt-4 text-center text-xs text-neutral-500">
          Your chat session will be preserved after signing in
        </div>
      </div>
    </div>
  );
}
