"use client";

import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { SkillFileViewer } from "@/components/contextual/SkillPackage/SkillFileViewer";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { SkillActions } from "../../../skills/[slug]/components/SkillActions";
import { SkillBody } from "../../../skills/[slug]/components/SkillBody";
import { useSkillPage } from "../../../skills/[slug]/components/useSkillPage";

interface Props {
  slug: string;
  onClose: () => void;
}

export function SkillDialog({ slug, onClose }: Props) {
  const {
    skill,
    isLoggedIn,
    isLoading,
    isError,
    refetch,
    isReady,
    isAdded,
    isAdding,
    experts,
    addToAutoPilot,
    addToExpert,
    files,
    openFilePath,
    openFile,
    closeFile,
  } = useSkillPage(slug);

  return (
    <>
      <Dialog
        title={skill?.title ?? "Skill"}
        styling={{ width: "760px" }}
        controlled={{
          isOpen: true,
          set: (open) => {
            if (!open) onClose();
          },
        }}
      >
        <Dialog.Content>
          <div data-testid="skill-dialog">
            {isLoading ? (
              <div role="status" aria-busy="true" className="space-y-2">
                <Skeleton className="h-4 w-full" />
                <Skeleton className="h-4 w-5/6" />
                <Skeleton className="h-4 w-2/3" />
              </div>
            ) : isError || !skill ? (
              <div className="flex items-center gap-2 text-sm text-zinc-600">
                <span>Couldn&apos;t load this skill right now.</span>
                <button
                  type="button"
                  onClick={() => refetch()}
                  className="font-medium text-accent underline-offset-2 transition-colors hover:underline"
                >
                  Retry
                </button>
              </div>
            ) : (
              <>
                <div className="mb-6 flex flex-wrap items-start justify-between gap-4 border-b border-zinc-200 pb-5">
                  <p className="min-w-0 max-w-[52ch] flex-1 text-[15px] leading-6 text-zinc-600">
                    {skill.description}
                  </p>
                  <SkillActions
                    slug={slug}
                    isLoggedIn={isLoggedIn}
                    isReady={isReady}
                    isAdded={isAdded}
                    isAdding={isAdding}
                    experts={experts}
                    onAdd={addToAutoPilot}
                    onAddToExpert={addToExpert}
                  />
                </div>
                {skill.body.trim() ? (
                  <SkillBody
                    body={skill.body}
                    title={skill.title}
                    packagePaths={files.map((file) => file.path)}
                    onOpenFile={openFile}
                  />
                ) : (
                  <p className="text-sm text-zinc-500">
                    This skill has no instructions yet.
                  </p>
                )}
              </>
            )}
          </div>
        </Dialog.Content>
      </Dialog>
      <SkillFileViewer
        source={{ kind: "listing", slug }}
        path={openFilePath}
        onClose={closeFile}
      />
    </>
  );
}
