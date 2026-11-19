"use client";

import { ArrowLeft01Icon, Calendar03Icon } from "@hugeicons/core-free-icons";
import { useRouter } from "next/navigation";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";

import { isServerFailure } from "../helpers";

interface Props {
  error: unknown;
  onRetry: () => void;
}

/** A run that cannot be read: deleted, a link that was never a run, or the
 *  server failing. Says so and offers a way back. */
export function RunNotFound({ error, onRetry }: Props) {
  const router = useRouter();
  const serverFailed = isServerFailure(error);
  return (
    <div className="mx-auto flex w-full max-w-md flex-col items-center gap-3 px-6 py-24 text-center">
      <Text variant="large-medium" as="h1" tone="ink">
        This run could not be found
      </Text>
      <Text variant="body" tone="quiet" className="text-balance">
        {serverFailed
          ? "The server did not answer. Try again in a moment."
          : "It may have been deleted, or the link is not right."}
      </Text>
      <div className="mt-2 flex flex-wrap items-center justify-center gap-2">
        <Button
          variant="secondary"
          size="small"
          leadingIcon={ArrowLeft01Icon}
          onClick={() => router.back()}
        >
          Go back
        </Button>
        <Button
          as="NextLink"
          href="/schedules"
          variant="secondary"
          size="small"
          leadingIcon={Calendar03Icon}
        >
          Open Schedules
        </Button>
        {serverFailed ? (
          <Button variant="primary" size="small" onClick={onRetry}>
            Try again
          </Button>
        ) : null}
      </div>
    </div>
  );
}
