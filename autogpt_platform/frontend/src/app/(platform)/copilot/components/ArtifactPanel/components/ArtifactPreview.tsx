import type { ComponentProps } from "react";
import { ArtifactContent } from "./ArtifactContent";

type Props = ComponentProps<typeof ArtifactContent> & {
  /** A written document sits straight on the panel, with no page under it. */
  isBare?: boolean;
};

export function ArtifactPreview({ isBare, ...props }: Props) {
  if (isBare) {
    return (
      <div className="flex min-h-0 flex-1 flex-col">
        <ArtifactContent {...props} />
      </div>
    );
  }
  return (
    <div className="flex min-h-0 flex-1 bg-surface-hover/70 p-3 sm:p-4">
      <div className="flex min-w-0 flex-1 flex-col overflow-hidden bg-surface shadow-sm">
        <ArtifactContent {...props} />
      </div>
    </div>
  );
}
