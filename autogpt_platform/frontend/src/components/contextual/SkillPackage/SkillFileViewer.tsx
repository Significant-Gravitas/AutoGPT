import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import {
  type SkillPackageSource,
  useSkillFileViewer,
} from "./useSkillFileViewer";

interface Props {
  source: SkillPackageSource;
  path: string | null;
  onClose: () => void;
}

export function SkillFileViewer({ source, path, onClose }: Props) {
  const { content, isLoading, unavailable } = useSkillFileViewer({
    source,
    path,
  });

  return (
    <Dialog
      title={<span className="break-all font-mono text-sm">{path}</span>}
      styling={{ width: "760px" }}
      controlled={{
        isOpen: path !== null,
        set: (open) => {
          if (!open) onClose();
        },
      }}
    >
      <Dialog.Content>
        <div data-testid="skill-file-viewer">
          {isLoading ? (
            <div className="space-y-2">
              <Skeleton className="h-4 w-full" />
              <Skeleton className="h-4 w-5/6" />
              <Skeleton className="h-4 w-2/3" />
            </div>
          ) : unavailable ? (
            <p className="text-sm text-foreground-tertiary">{unavailable}</p>
          ) : (
            <pre className="max-h-[60vh] overflow-auto rounded-md bg-surface-muted p-3 font-mono text-body-sm leading-5 text-foreground-body">
              {content}
            </pre>
          )}
        </div>
      </Dialog.Content>
    </Dialog>
  );
}
