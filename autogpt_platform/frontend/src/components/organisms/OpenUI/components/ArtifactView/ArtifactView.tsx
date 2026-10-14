import { useState } from "react";
import type { ComponentRenderProps } from "@openuidev/react-lang";
import type { z } from "zod/v4";
import type { Artifact } from "@/lib/openui/catalog-artifacts";
import { workspaceFileRenderer } from "@/components/contextual/OutputRenderers/renderers/WorkspaceFileRenderer";
import { imageRenderer } from "@/components/contextual/OutputRenderers/renderers/ImageRenderer";
import { videoRenderer } from "@/components/contextual/OutputRenderers/renderers/VideoRenderer";
import { audioRenderer } from "@/components/contextual/OutputRenderers/renderers/AudioRenderer";
import { Button } from "@/components/atoms/Button/Button";
import { useCopilotChatActions } from "@/app/(platform)/copilot/components/CopilotChatActionsProvider/useCopilotChatActions";

export function ArtifactView({
  props,
}: ComponentRenderProps<z.infer<typeof Artifact.props>>) {
  const { chatSurface } = useCopilotChatActions();
  const [failed, setFailed] = useState(false);
  const [attempt, setAttempt] = useState(0);
  const workspace = props.uri.startsWith("workspace://");
  const unavailable = workspace && chatSurface === "share";
  const metadata = { filename: props.filename, mimeType: props.mimeType };
  const renderer = workspace
    ? workspaceFileRenderer
    : props.mimeType.startsWith("image/")
      ? imageRenderer
      : props.mimeType.startsWith("video/")
        ? videoRenderer
        : props.mimeType.startsWith("audio/")
          ? audioRenderer
          : null;
  const download =
    renderer?.getDownloadContent(props.uri, metadata)?.data ?? props.uri;
  return (
    <figure className="min-w-0 space-y-3" aria-label={props.title}>
      <figcaption>
        <h3 className="font-semibold">{props.title}</h3>
        <p className="text-sm text-muted-foreground">{props.description}</p>
      </figcaption>
      {unavailable ? (
        <p role="status" className="text-sm text-muted-foreground">
          This workspace file is private. Open the original conversation to view
          it.
        </p>
      ) : (
        <>
          {failed ? (
            <div role="status" className="space-y-2 text-sm">
              <p>
                This file could not load. The link may have expired or access
                may have changed.
              </p>
              <Button
                type="button"
                variant="secondary"
                size="small"
                onClick={() => {
                  setFailed(false);
                  setAttempt(attempt + 1);
                }}
              >
                Retry preview
              </Button>
            </div>
          ) : (
            <div key={attempt} onErrorCapture={() => setFailed(true)}>
              {renderer?.render(props.uri, {
                ...metadata,
                filename: props.mimeType.startsWith("image/")
                  ? props.description || props.filename
                  : props.filename,
              })}
            </div>
          )}
          {typeof download === "string" && (
            <a
              className="inline-block min-h-11 py-3 text-sm text-primary underline"
              href={download}
              download={props.filename}
              target="_blank"
              rel="noopener noreferrer"
            >
              Download {props.filename}
            </a>
          )}
        </>
      )}
    </figure>
  );
}
