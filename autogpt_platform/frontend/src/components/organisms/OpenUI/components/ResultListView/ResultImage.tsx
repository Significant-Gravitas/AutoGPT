import Image from "next/image";
import { useState } from "react";
import { Button } from "@/components/atoms/Button/Button";
import { workspaceFileRenderer } from "@/components/contextual/OutputRenderers/renderers/WorkspaceFileRenderer";
import { useCopilotChatActions } from "@/app/(platform)/copilot/components/CopilotChatActionsProvider/useCopilotChatActions";

interface Props {
  image: { url: string; alt: string };
}
export function ResultImage({ image }: Props) {
  const { chatSurface } = useCopilotChatActions();
  const [failed, setFailed] = useState(false);
  const workspace = image.url.startsWith("workspace://");
  if (workspace && chatSurface === "share")
    return (
      <p className="text-xs text-muted-foreground">
        This image is private to the original conversation.
      </p>
    );
  const url = workspace
    ? workspaceFileRenderer.getDownloadContent(image.url)?.data
    : image.url;
  if (typeof url !== "string") return null;
  if (failed)
    return (
      <div className="text-xs text-muted-foreground">
        <p>Image unavailable. The link or permission may have changed.</p>
        <Button
          type="button"
          size="small"
          variant="ghost"
          onClick={() => setFailed(false)}
        >
          Retry {image.alt}
        </Button>
      </div>
    );
  return (
    <Image
      src={url}
      alt={image.alt}
      width={480}
      height={240}
      unoptimized
      onError={() => setFailed(true)}
      className="h-36 w-full rounded-lg object-contain"
    />
  );
}
