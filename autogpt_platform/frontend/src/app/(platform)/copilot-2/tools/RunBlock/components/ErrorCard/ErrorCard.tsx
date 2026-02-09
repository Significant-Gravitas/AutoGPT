"use client";

import type { ErrorResponse } from "@/app/api/__generated__/models/errorResponse";
import { formatMaybeJson } from "../../helpers";

interface Props {
  output: ErrorResponse;
}

export function ErrorCard({ output }: Props) {
  return (
    <div className="grid gap-2">
      <p className="text-sm text-foreground">{output.message}</p>
      {output.error && (
        <pre className="whitespace-pre-wrap rounded-2xl border bg-muted/30 p-3 text-xs text-muted-foreground">
          {formatMaybeJson(output.error)}
        </pre>
      )}
      {output.details && (
        <pre className="whitespace-pre-wrap rounded-2xl border bg-muted/30 p-3 text-xs text-muted-foreground">
          {formatMaybeJson(output.details)}
        </pre>
      )}
    </div>
  );
}
