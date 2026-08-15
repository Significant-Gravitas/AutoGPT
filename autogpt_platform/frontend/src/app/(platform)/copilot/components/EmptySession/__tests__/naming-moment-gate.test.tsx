import { render, screen } from "@/tests/integrations/test-utils";
import { beforeEach, expect, test, vi } from "vitest";
import { EmptySession } from "../EmptySession";

const { introMock } = vi.hoisted(() => ({
  introMock: {
    current: {
      isVisible: false,
      isAwaitingGreeting: false,
      anchorTop: false,
      isWelcomeOpen: false,
      closeWelcome: vi.fn(),
      greeting: "",
      prompts: [],
      transcript: "",
      path: "B",
    },
  },
}));

vi.mock(
  "@/app/(platform)/copilot/components/OnboardingIntroCard/useOnboardingIntroCard",
  () => ({ useOnboardingIntroCard: () => introMock.current }),
);

vi.mock(
  "@/app/(platform)/copilot/components/NamingMomentCard/NamingMomentCard",
  () => ({ NamingMomentCard: () => <div>Naming moment prompt</div> }),
);

vi.mock("@/app/api/__generated__/endpoints/chat/chat", () => ({
  useGetV2GetSuggestedPrompts: () => ({ data: undefined, isLoading: false }),
}));

vi.mock("@/app/(platform)/copilot/components/ChatInput/ChatInput", () => ({
  ChatInput: () => null,
}));

vi.mock(
  "@/app/(platform)/copilot/components/EmptySession/useRecipientPicker",
  () => ({
    useRecipientPicker: () => ({
      options: [],
      recipient: null,
      isLoadingRecipient: false,
      selectRecipient: vi.fn(),
    }),
  }),
);

vi.mock("@/app/(platform)/copilot/components/PulseChips/usePulseChips", () => ({
  usePulseChips: () => [],
}));

vi.mock("@/services/feature-flags/use-get-flag", async (importActual) => {
  const actual =
    await importActual<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return { ...actual, useGetFlag: () => false };
});

vi.mock("@/components/ui/dot-distortion-shader", () => ({
  DotDistortionShader: () => null,
}));

const props = {
  inputLayoutId: "test-layout",
  isCreatingSession: false,
  onCreateSession: vi.fn(),
  isUploadingFiles: false,
  droppedFiles: [],
  onDroppedFilesConsumed: vi.fn(),
  isAdoptingExpertSession: false,
  onSend: vi.fn(),
};

beforeEach(() => {
  introMock.current = {
    ...introMock.current,
    isVisible: false,
    isAwaitingGreeting: false,
  };
});

test.each([
  { isVisible: true, isAwaitingGreeting: false },
  { isVisible: false, isAwaitingGreeting: true },
])(
  "hides the naming moment during the greeting flow",
  ({ isVisible, isAwaitingGreeting }) => {
    introMock.current = {
      ...introMock.current,
      isVisible,
      isAwaitingGreeting,
    };

    render(<EmptySession {...props} />);

    expect(screen.queryByText("Naming moment prompt")).toBeNull();
  },
);

test("shows the naming moment after the greeting flow is inactive", () => {
  render(<EmptySession {...props} />);

  expect(screen.getByText("Naming moment prompt")).toBeDefined();
});
