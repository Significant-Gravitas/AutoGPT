import type { ComponentProps } from "react";
import { ArtifactContent } from "./ArtifactContent";

export function ArtifactPreview(props: ComponentProps<typeof ArtifactContent>) {
  return (
    <div className="flex min-h-0 flex-1 bg-muted/70 p-3 sm:p-4">
      <div className="flex min-w-0 flex-1 flex-col overflow-hidden bg-card shadow-sm">
        <ArtifactContent {...props} />
      </div>
    </div>
  );
}
