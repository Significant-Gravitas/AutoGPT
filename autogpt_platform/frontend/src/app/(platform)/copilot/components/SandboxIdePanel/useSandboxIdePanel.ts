import { useCopilotUIStore } from "../../store";

export function useSandboxIdePanel() {
  const panel = useCopilotUIStore((s) => s.sandboxIdePanel);
  const setActiveTab = useCopilotUIStore((s) => s.setSandboxIdeTab);
  const close = useCopilotUIStore((s) => s.closeSandboxIdePanel);
  const selectFile = useCopilotUIStore((s) => s.selectSandboxFile);
  const setWidth = useCopilotUIStore((s) => s.setSandboxIdeWidth);

  return {
    isOpen: panel.isOpen,
    activeTab: panel.activeTab,
    selectedFilePath: panel.selectedFilePath,
    width: panel.width,
    setActiveTab,
    close,
    selectFile,
    setWidth,
  };
}
