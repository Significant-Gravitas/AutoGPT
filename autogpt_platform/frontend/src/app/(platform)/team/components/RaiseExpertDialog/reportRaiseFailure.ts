import { toast } from "@/components/molecules/Toast/use-toast";
import { ApiError } from "@/lib/autogpt-server-api/helpers";

export function reportRaiseFailure(error: unknown, name: string) {
  const code = limitCode(error);
  if (code === "raised_expert_lifetime_limit") {
    toast({
      title: "Expert creation limit reached",
      description:
        "This account has reached its lifetime limit for created Experts. Contact support if you need more capacity.",
      variant: "destructive",
    });
    return;
  }
  if (code === "active_expert_limit") {
    toast({
      title: "Your team is full",
      description:
        "You've reached the limit of active experts. Archive one from your team page to create another.",
      variant: "destructive",
    });
    return;
  }
  toast({
    title: `Couldn't create ${name}`,
    description:
      error instanceof ApiError && [400, 402, 422, 503].includes(error.status)
        ? error.message
        : "Something went wrong. Please try again.",
    variant: "destructive",
  });
}

function limitCode(error: unknown): string | null {
  if (!(error instanceof ApiError) || error.status !== 409) return null;
  const response: unknown = error.response;
  if (!response || typeof response !== "object" || !("detail" in response)) {
    return "active_expert_limit";
  }
  const detail = response.detail;
  if (detail && typeof detail === "object" && "code" in detail) {
    return typeof detail.code === "string" ? detail.code : null;
  }
  return "active_expert_limit";
}
