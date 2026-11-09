import { ApiError } from "@/lib/autogpt-server-api/helpers";

// A reset waits for the expert's running work: a turn that ends mid-reset
// would put what it learned back into the memory the reset just erased.
export function resetErrorToast(error: unknown, expertName: string) {
  if (error instanceof ApiError && error.status === 409) {
    return {
      title: `${expertName} is still working`,
      description: `Reset ${expertName} once they finish.`,
    };
  }
  return {
    title: `Could not finish resetting ${expertName}`,
    description: "Please try again.",
    variant: "destructive" as const,
  };
}
