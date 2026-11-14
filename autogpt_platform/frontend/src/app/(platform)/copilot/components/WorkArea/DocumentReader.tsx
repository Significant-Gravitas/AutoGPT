"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import { cn } from "@/lib/utils";
import {
  Cancel01Icon,
  Download01Icon,
  File02Icon,
} from "@hugeicons/core-free-icons";
import * as Dialog from "@radix-ui/react-dialog";
import { useRef } from "react";
import type { ArtifactRef } from "../../store";
import { ArtifactContent } from "../ArtifactPanel/components/ArtifactContent";
import { getCachedArtifactContent } from "../ArtifactPanel/components/useArtifactContent";
import type { ArtifactClassification } from "../ArtifactPanel/helpers";
import { FOCUS } from "../EmptySession/components/ExpertInfoPanel/components/InfoParts";
import { CopyTool, Tool, ViewSwitch } from "./artifactTools";
import { headingFor, outlineOf } from "./documentOutline";

const READER_PROSE = "mx-auto w-full max-w-[64ch] px-8 pt-6 pb-12";
const MIN_OUTLINE = 2;

interface Props {
  artifact: ArtifactRef;
  classification: ArtifactClassification;
  isSourceView: boolean;
  onSourceView: (isSource: boolean) => void;
  canCopy: boolean;
  onCopy: () => void;
  onDownload: () => void;
  onClose: () => void;
}

/** A written document lifted over the chat to be read properly: a wide
 *  sheet with a comfortable line length and its sections down the side.
 *  Esc, the cross or a click outside puts it back in the panel. */
export function DocumentReader({
  artifact,
  classification,
  isSourceView,
  onSourceView,
  canCopy,
  onCopy,
  onDownload,
  onClose,
}: Props) {
  const page = useRef<HTMLDivElement>(null);
  const outline =
    classification.type === "markdown" && !isSourceView
      ? outlineOf(getCachedArtifactContent(artifact.id) ?? "")
      : [];

  return (
    <Dialog.Root open onOpenChange={(isOpen) => !isOpen && onClose()}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-50 bg-foreground-ink/20" />
        <Dialog.Content
          aria-describedby={undefined}
          className="fixed inset-4 z-50 mx-auto flex max-w-[58rem] animate-reader-in flex-col overflow-hidden rounded-panel bg-surface shadow-(--inbox-shadow-card-hover) outline-none md:inset-6"
        >
          <div className="flex h-11 shrink-0 items-center gap-1 px-3 shadow-(--inbox-shadow-hairline-b)">
            <Icon icon={File02Icon} size={14} tone="quiet" />
            <Dialog.Title className="mr-2 min-w-0 truncate text-body-sm font-medium text-foreground-ink">
              {artifact.title}
            </Dialog.Title>
            {classification.hasSourceToggle ? (
              <ViewSwitch isSource={isSourceView} onChange={onSourceView} />
            ) : null}
            <span className="flex-1" />
            {canCopy ? <CopyTool isLabelled onCopy={onCopy} /> : null}
            <Tool
              icon={Download01Icon}
              label="Download"
              isLabelled
              onClick={onDownload}
            />
            <Dialog.Close
              aria-label="Close"
              title="Close (Esc)"
              className={cn(
                "flex size-7 shrink-0 items-center justify-center rounded-md text-foreground-quiet transition-colors duration-100 ease-out hover:bg-foreground-ink/5 hover:text-foreground-ink",
                FOCUS,
              )}
            >
              <Icon icon={Cancel01Icon} size={14} tone="inherit" />
            </Dialog.Close>
          </div>
          <div className="flex min-h-0 flex-1">
            {outline.length >= MIN_OUTLINE ? (
              <nav
                aria-label="Sections"
                className="hidden w-52 shrink-0 flex-col gap-0.5 overflow-y-auto p-3 md:flex"
              >
                {outline.map((entry, position) => (
                  <button
                    key={`${position}:${entry.text}`}
                    type="button"
                    title={entry.text}
                    onClick={() =>
                      headingFor(page.current, entry, position)?.scrollIntoView(
                        { behavior: "smooth", block: "start" },
                      )
                    }
                    className={cn(
                      "truncate rounded-md py-1 pr-2 text-left text-body-sm text-foreground-quiet transition-colors duration-100 ease-out hover:bg-surface-tray hover:text-foreground-ink",
                      entry.level === 1 && "pl-2 font-medium",
                      entry.level === 2 && "pl-2",
                      entry.level === 3 && "pl-5",
                      FOCUS,
                    )}
                  >
                    {entry.text}
                  </button>
                ))}
              </nav>
            ) : null}
            <div ref={page} className="flex min-h-0 min-w-0 flex-1 flex-col">
              <ArtifactContent
                artifact={artifact}
                isSourceView={isSourceView}
                classification={classification}
                prose={READER_PROSE}
              />
            </div>
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
