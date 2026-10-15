// Harness-only behavior fixtures. All visible composer components remain
// imported unchanged from the PR checkout. Null components below
// correspond to empty/closed UI in this fixture and are never visible.
import React, { useRef } from "react";
import { fixture } from "./fixtures";

const noop = () => {};
export const Flag = new Proxy({}, { get: (_, key) => key });
export function useGetFlag() {
  return false;
}
export function toast() {}
export function isTokenDevtoolEnabled() {
  return false;
}
export function useCopilotUIStore() {
  return {
    isDryRun: false,
    setIsDryRun: noop,
    initialPrompt: null,
    setInitialPrompt: noop,
    notifyMessageSent: noop,
  };
}
export function useAuthStore<T>(
  selector: (state: { user: { id: string } }) => T,
): T {
  return selector({ user: { id: "fixture-user" } });
}
export function useChatMentions() {
  return {
    isOpen: false,
    bindInput: noop,
    detect: noop,
    onKeyDown: () => false,
    close: noop,
  };
}
export function useConnectedIntegrations() {
  return [];
}
export function useOnboardingMicGlow() {
  return { isGlowing: false, dismissGlow: noop };
}
export function useVoiceRecording({ disabled }: { disabled?: boolean }) {
  return {
    isRecording: false,
    isTranscribing: false,
    transcriptionError: null,
    hasFailedRecording: false,
    retryTranscription: noop,
    downloadFailedRecording: noop,
    dismissTranscriptionError: noop,
    elapsedTime: 0,
    toggleRecording: noop,
    handleKeyDown: noop,
    showMicButton: true,
    isInputDisabled: disabled,
    audioStream: null,
  };
}
export function useComposerPlusMenu() {
  const fileInputRef = useRef<HTMLInputElement>(null);
  return {
    fileInputRef,
    openModal: noop,
    showWorkspaceOption: false,
    openFilePicker: noop,
    handleFileChange: noop,
  };
}
const platformOffer = {
  offer_id: "platform:deployment",
  provider_family: "autogpt",
  display_name: "AutoGPT Platform",
  auth_method: "deployment",
  auth_provider: "platform",
  credential_id: null,
  backed_by_label: "Your AutoGPT plan",
  description: "New chats are backed by your AutoGPT plan.",
  state: "ready",
  selectable: true,
  is_default: true,
  limitations: [],
  tiers: [
    {
      tier: "standard",
      label: "Balanced",
      selectable: true,
      display_model: null,
    },
    {
      tier: "advanced",
      label: "Advanced",
      selectable: false,
      display_model: null,
      lock_reason: "Upgrade your plan to use Advanced.",
    },
  ],
};
export function useConnectionPicker() {
  return {
    offers: [platformOffer],
    active: platformOffer,
    chooseConnection: noop,
    tier: "standard",
    setTier: noop,
    showTiers: true,
    hasConnectionChoice: false,
    connectChatGPT: noop,
    isConnecting: false,
    canConnectChatGPT: false,
    isLoading: false,
    isError: false,
  };
}
export function FileChips() {
  return null;
}
export function MentionDropdown() {
  return null;
}
export function WorkspaceFilePicker() {
  return null;
}
export function RecordingIndicator() {
  return null;
}
export function TranscriptionErrorBar() {
  return null;
}
export function TokenDevtoolBadge() {
  return null;
}
export function DryRunToggleButton() {
  return null;
}
export function ChoiceRow() {
  return null;
}
export function ConnectAccountRow() {
  return null;
}
export function MaxUpgradeCard() {
  return null;
}
export function TierToggle() {
  return null;
}
export default function Image(
  props: React.ImgHTMLAttributes<HTMLImageElement>,
) {
  return <img {...props} />;
}

export function useProviderLimitDialog() {
  return {
    alternative: fixture.linkedProvider ? { display_name: "Claude" } : null,
    continueHere: noop,
    isSwitching: false,
    resetHint: null,
    isLoadingOffers: false,
    failedToLoadOffers: false,
    retryOffers: noop,
  };
}

export function useFlagStatus() {
  return { enabled: true, ready: true };
}
