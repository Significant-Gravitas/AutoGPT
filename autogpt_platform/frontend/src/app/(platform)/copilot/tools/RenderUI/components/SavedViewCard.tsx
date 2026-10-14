import { OpenUIFileView } from "../../../components/OpenUIFileView/OpenUIFileView";
import { openUIArtifactRef } from "../../../components/OpenUIFileView/helpers";
import { useCopilotUIStore } from "../../../store";
import { defaultViewFileName } from "@/lib/openui/file";
import type { SavedFileRef } from "../helpers";
import { StaleRevision } from "../../../components/OpenUIFileView/components/StaleRevision";

interface Props {
  file: SavedFileRef;
  sessionId: string | null;
  source: string;
  title: string;
}

export function SavedViewCard({ file, sessionId, source, title }: Props) {
  const openArtifact = useCopilotUIStore((state) => state.openArtifact);
  const name = file.name ?? defaultViewFileName(title);
  const header = {
    name,
    onOpen: () => openArtifact(openUIArtifactRef(file.id, name)),
  };
  return (
    <OpenUIFileView
      fileId={file.id}
      currentSessionId={sessionId}
      cardSource={source}
      header={header}
      fallback={
        <StaleRevision
          source={source}
          notice="This view's file is no longer available. It is shown read-only."
        />
      }
    />
  );
}
