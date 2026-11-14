import { TASK_PARAM } from "@/app/(platform)/tasks/helpers";
import { useListExperts } from "@/app/api/__generated__/endpoints/experts/experts";
import { useQueryState } from "nuqs";
import { useEffect, useRef } from "react";
import {
  DEFAULT_PANEL_WIDTH,
  PANEL_RESERVED_WIDTH,
  useCopilotUIStore,
  type ContextPanelTab,
} from "../../store";
import {
  artifactTabKey,
  COMPUTER_TAB_KEY,
  CONTEXT_TAB_KEY,
  useWorkAreaStore,
} from "../../workArea";

const CONTEXT_TITLES: Record<ContextPanelTab, string> = {
  files: "Files",
  artifacts: "Artifacts",
  integrations: "Integrations",
};

/** The two widths the sidebar's button moves between: a glance beside the
 *  chat, and enough room to read a document. Dragging its edge sets any
 *  width in between. */
const NARROW_WIDTH = DEFAULT_PANEL_WIDTH;
const WIDE_WIDTH = 760;
const WIDE_FROM = 560;
/** A tab's name until its task is read. */
const UNREAD_TITLE = "Task";

/** The right sidebar's tabs: the tasks and files opened in this chat, plus
 *  the Files or Integrations face while the chat has it open. The artifact
 *  the store shows always has a tab, and is the one in front. */
export function useWorkArea(sessionID: string | null) {
  const tabs = useWorkAreaStore((s) => s.tabs);
  const storedKey = useWorkAreaStore((s) => s.activeKey);
  const showArtifact = useWorkAreaStore((s) => s.showArtifact);
  const activate = useWorkAreaStore((s) => s.activate);
  const closeTab = useWorkAreaStore((s) => s.close);
  const reset = useWorkAreaStore((s) => s.reset);
  const isFolded = useWorkAreaStore((s) => s.isFolded);
  const unfold = useWorkAreaStore((s) => s.unfold);
  const openStart = useWorkAreaStore((s) => s.openStart);
  const openTask = useWorkAreaStore((s) => s.openTask);
  const reportTabCount = useWorkAreaStore((s) => s.reportTabCount);
  const toggleContextTab = useCopilotUIStore((s) => s.toggleContextPanelTab);
  const openComputer = useCopilotUIStore((s) => s.openComputer);
  const toggleIntegrations = useCopilotUIStore(
    (s) => s.toggleIntegrationsPanel,
  );
  const contextExpert = useCopilotUIStore((s) => s.contextPanelExpert);

  const panel = useCopilotUIStore((s) => s.artifactPanel);
  const openArtifact = useCopilotUIStore((s) => s.openArtifact);
  const closeArtifactPanel = useCopilotUIStore((s) => s.closeArtifactPanel);
  const closeComputer = useCopilotUIStore((s) => s.closeComputer);
  const setPanelMode = useCopilotUIStore((s) => s.setArtifactPanelMode);
  const width = useCopilotUIStore((s) => s.artifactPanelWidth);
  const setWidth = useCopilotUIStore((s) => s.setArtifactPanelWidth);
  const isWide = width >= WIDE_FROM;

  function toggleWidth() {
    if (isWide) return setWidth(NARROW_WIDTH);
    // Never wider than what leaves the chat its own room.
    const room = window.innerWidth - PANEL_RESERVED_WIDTH;
    setWidth(Math.max(NARROW_WIDTH, Math.min(WIDE_WIDTH, room)));
  }

  const experts = useListExperts({
    query: { select: (res) => (res.status === 200 ? res.data : []) },
  });

  // Another chat has its own tabs.
  useEffect(() => reset, [sessionID, reset]);

  // A link to a task lands on its chat with the task open beside it.
  const [linkedTask] = useQueryState(TASK_PARAM);
  useEffect(() => {
    if (linkedTask) openTask(linkedTask, UNREAD_TITLE);
  }, [sessionID, linkedTask, openTask]);

  // Whatever opens an artifact (a chip, the files list, Otto) goes through
  // the store; its tab follows.
  const artifact = panel.activeArtifact;
  // The chat's own faces need a chat; Home has only tasks and their files.
  const isChat = sessionID !== null;
  const hasComputer =
    isChat && panel.isComputerOpen && panel.mode === "computer";
  useEffect(() => {
    if (artifact) showArtifact(artifact);
  }, [artifact, showArtifact]);

  const hasContext = isChat && panel.isOpen && !artifact && !hasComputer;
  useEffect(() => {
    if (!hasContext) return;
    activate(CONTEXT_TAB_KEY);
    unfold();
  }, [hasContext, panel.activeTab, activate, unfold]);

  const artifactKey = artifact
    ? artifactTabKey(artifact.documentId ?? artifact.id)
    : null;
  // The computer comes forward when the chat turns the panel to it.
  useEffect(() => {
    if (!hasComputer) return;
    activate(COMPUTER_TAB_KEY);
    unfold();
  }, [hasComputer, activate, unfold]);

  const keys = [
    ...(hasContext ? [CONTEXT_TAB_KEY] : []),
    ...tabs.map((tab) => tab.key),
    ...(isChat && panel.isComputerOpen ? [COMPUTER_TAB_KEY] : []),
  ];
  const activeKey =
    storedKey && keys.includes(storedKey)
      ? storedKey
      : (keys[keys.length - 1] ?? null);

  // The viewer shows the store's artifact, so the two are kept in step: a
  // file the store closed loses its tab, and a file tab that comes forward
  // (its neighbour was closed) is put back in the store.
  const previousArtifactKey = useRef(artifactKey);
  useEffect(() => {
    const was = previousArtifactKey.current;
    previousArtifactKey.current = artifactKey;
    if (was && !artifactKey) {
      closeTab(was);
      return;
    }
    const front = tabs.find((tab) => tab.key === activeKey);
    if (front?.kind === "artifact" && front.key !== artifactKey) {
      openArtifact(front.artifact);
    }
  }, [artifactKey, activeKey, tabs, closeTab, openArtifact]);

  // The open button needs to know whether there is anything to come back to.
  const tabCount = keys.length;
  useEffect(() => {
    reportTabCount(tabCount);
  }, [tabCount, reportTabCount]);

  // What the "This chat" tab can open beside itself.
  function openFromOverview(what: "files" | "computer" | "integrations") {
    if (what === "computer") openComputer();
    else if (what === "integrations" && contextExpert) {
      toggleIntegrations(contextExpert);
    } else if (!(hasContext && panel.activeTab === "files")) {
      toggleContextTab("files");
    }
  }

  function select(key: string) {
    const tab = tabs.find((entry) => entry.key === key);
    if (key === COMPUTER_TAB_KEY) setPanelMode("computer");
    // The viewer shows the store's artifact, so coming back to a file's tab
    // puts that file back in the store.
    if (tab?.kind === "artifact") {
      setPanelMode("artifact");
      if (key !== artifactKey) openArtifact(tab.artifact);
    }
    activate(key);
  }

  function close(key: string) {
    if (key === CONTEXT_TAB_KEY) {
      closeArtifactPanel();
      return;
    }
    if (key === COMPUTER_TAB_KEY) {
      closeComputer();
      return;
    }
    closeTab(key);
    if (key !== artifactKey) return;
    const other = tabs.findLast(
      (tab) => tab.kind === "artifact" && tab.key !== key,
    );
    if (other?.kind === "artifact") openArtifact(other.artifact);
    else closeArtifactPanel();
  }

  return {
    tabs,
    hasContext,
    contextTitle: CONTEXT_TITLES[panel.activeTab],
    activeKey,
    isOpen: keys.length > 0 && !isFolded,
    openStart,
    openFromOverview,
    openTask,
    hasExpert: contextExpert !== null,
    hasComputer: isChat && panel.isComputerOpen,
    contextTab: panel.activeTab,
    openArtifact,
    experts: experts.data ?? [],
    width,
    setWidth,
    isWide,
    toggleWidth,
    select,
    close,
  };
}
