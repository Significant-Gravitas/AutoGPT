"use client";

import { useRouter } from "next/navigation";
import { Button } from "@/components/atoms/Button/Button";
import type { ExecutionStartedResponse } from "@/app/api/__generated__/models/executionStartedResponse";

interface Props {
  output: ExecutionStartedResponse;
}

export function ExecutionStartedCard({ output }: Props) {
  const router = useRouter();

  return (
    <div className="grid gap-2">
      <div className="rounded-2xl border bg-background p-3">
        <div className="min-w-0">
          <p className="text-sm font-medium text-foreground">
            Execution started
          </p>
          <p className="mt-0.5 truncate text-xs text-muted-foreground">
            {output.execution_id}
          </p>
          <p className="mt-2 text-xs text-muted-foreground">{output.message}</p>
        </div>
        {output.library_agent_link && (
          <Button
            variant="outline"
            size="small"
            className="mt-3 w-full"
            onClick={() => router.push(output.library_agent_link!)}
          >
            View Execution
          </Button>
        )}
      </div>
    </div>
  );
}
