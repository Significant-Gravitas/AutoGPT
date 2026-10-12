import path from "node:path";
const root = path.dirname(new URL(import.meta.url).pathname);
const mocks = path.join(root, "composer-mocks.tsx");

// Register before generic component-source path resolvers. This plugin
// replaces only behavior hooks and components invisible in the fixture;
// ChatInput, its textarea, plus/microphone/tier/send buttons stay original.
export const composerPlugin = {
  name: "component-composer-fixtures",
  setup(build) {
    build.onResolve(
      {
        filter:
          /(^next\/image$|\/use-get-flag$|\/use-toast$|\/copilot\/store$|^(\.\.\/)+store$|\/tokenDevtool\/gate$|\/(useProviderLimitDialog|useAuthStore|useChatMentions|useConnectedIntegrations|useVoiceRecording|useOnboardingMicGlow|useComposerPlusMenu|useConnectionPicker|FileChips|MentionDropdown|WorkspaceFilePicker|RecordingIndicator|TranscriptionErrorBar|TokenDevtoolBadge|DryRunToggleButton|ChoiceRow|ConnectAccountRow|MaxUpgradeCard|TierToggle)$)/,
      },
      () => ({ path: mocks }),
    );
  },
};
export default composerPlugin;
