import { createContext, useContext } from "react";
import type { ArtifactRef } from "../../../store";
import { useCopilotUIStore } from "../../../store";
import { classifyArtifact } from "../../ArtifactPanel/helpers";
import { defaultWorkspaceFileUrl } from "../helpers";

interface WorkspaceLinks {
  artifacts: ArtifactRef[];
  readOnly?: boolean;
  fileURL: (fileID: string) => string;
}

/** The message's files, for the links its markdown draws. */
export const WorkspaceLinksContext = createContext<WorkspaceLinks>({
  artifacts: [],
  fileURL: defaultWorkspaceFileUrl,
});

interface Props {
  fileID: string;
  label?: string;
}

/** A link to a file in a chat reply: a plain click opens it in the panel, as
 *  its card does; any other click, and the share view, keep the download. */
export function WorkspaceLink({ fileID, label }: Props) {
  const { artifacts, readOnly, fileURL } = useContext(WorkspaceLinksContext);
  const openArtifact = useCopilotUIStore((s) => s.openArtifact);
  // A file the message also draws as an image has no card, so only a download.
  const artifact = artifacts.find((candidate) => candidate.id === fileID);

  function handleClick(event: React.MouseEvent<HTMLAnchorElement>) {
    if (!artifact || readOnly || event.button !== 0) return;
    if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey)
      return;
    const { openable } = classifyArtifact(
      artifact.mimeType,
      artifact.title,
      artifact.sizeBytes,
    );
    if (!openable) return;
    event.preventDefault();
    openArtifact(artifact);
  }

  return (
    <a
      href={artifact?.sourceUrl ?? fileURL(fileID)}
      target="_blank"
      rel="noreferrer"
      onClick={handleClick}
      data-streamdown="link"
      // Streamdown's own link look; doubled to outrank the chat's `[&_a]` colour.
      className="wrap-anywhere font-medium [&&]:text-primary [&&]:underline"
    >
      {label}
    </a>
  );
}
