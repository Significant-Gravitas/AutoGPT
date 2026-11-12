import type { Meta, StoryObj } from "@storybook/nextjs";
import { ProgressBar } from "../../../components/ProgressBar";
import { StepIndicator } from "../../../components/StepIndicator";
import OnboardingLayout from "../../../layout";
import { PAYWALL_LAST_STEPS } from "../../../store";
import { InsufficientState } from "./InsufficientState";
import { type InsufficientMode, type OttoReplyKind } from "./insufficient-copy";

// The hosted wizard's step count, as useOnboardingLayout derives it.
const TOTAL_STEPS = PAYWALL_LAST_STEPS.preparing - 1;
const STEP = PAYWALL_LAST_STEPS.painPoints!;

interface Args {
  mode: InsufficientMode;
  heardText: string | null;
  replyKind: OttoReplyKind | null;
}

// The not-enough screen in the onboarding page's chrome, at the width the
// brain dump step gives it, so each of Otto's lines can be checked where it
// really sits.
function NotEnoughScreen({ mode, heardText, replyKind }: Args) {
  return (
    <div className="fixed inset-0 overflow-auto">
      <OnboardingLayout>
        <div className="flex min-h-screen w-full flex-col items-center">
          <ProgressBar currentStep={STEP} totalSteps={TOTAL_STEPS} />
          <div className="flex w-full min-w-0 flex-1 items-center justify-center pt-16 pb-8">
            <div className="relative z-10 flex w-full max-w-2xl flex-col items-center gap-6 px-4">
              <InsufficientState
                mode={mode}
                heardText={heardText}
                replyKind={replyKind}
                role="Founder/CEO"
                canRecord
                onTalk={() => {}}
                onWrite={() => {}}
                onSkip={() => {}}
              />
            </div>
          </div>
          <div className="pb-8">
            <StepIndicator totalSteps={TOTAL_STEPS} currentStep={STEP} />
          </div>
        </div>
      </OnboardingLayout>
    </div>
  );
}

const meta = {
  title: "Onboarding/Brain dump/Not enough",
  parameters: { layout: "fullscreen" },
  render: (args) => <NotEnoughScreen {...args} />,
} satisfies Meta<Args>;

export default meta;
type Story = StoryObj<typeof meta>;

function voice(heardText: string, replyKind: OttoReplyKind | null): Story {
  return { args: { mode: "voice", heardText, replyKind } };
}

function typed(heardText: string, replyKind: OttoReplyKind | null): Story {
  return { args: { mode: "typed", heardText, replyKind } };
}

export const MicCheck = voice("Um, okay. Hi. Is this working?", "mic_check");
export const Greeting = voice("Hi Otto, nice to meet you.", "greeting");
export const Unsure = voice("Hmm. I don't know.", "unsure");
export const Reluctant = voice("I don't want to do this.", "reluctant");
export const WhatToSay = voice("What am I supposed to say?", "what_to_say");
export const WantsToSkip = voice("Can I skip this?", "wants_to_skip");
export const AboutOtto = voice("What is this?", "about_otto");
export const Busy = voice("Hang on, one second.", "busy");
export const Annoyed = voice("This is pointless.", "annoyed");
export const SomeWork = voice("I do marketing.", "some_work");
export const Garbled = voice("the the window and", "garbled");
export const VoiceOther = voice("I like pizza.", "other");
export const VoiceNoPick = voice("Hello there.", null);
export const NoWords: Story = {
  args: { mode: "voice", heardText: null, replyKind: null },
};
export const TypedTest = typed("asdf", "test");
export const TypedSkip = typed("skip", "wants_to_skip");
export const TypedNoPick = typed("idk", null);
// The longest line under the longest "You" bubble (three lines, clamped).
export const LongestTake = voice(
  "Okay so I'm just going to keep talking for a while here because I'm not " +
    "really sure what you want me to say, and honestly I'm still thinking " +
    "about it, so give me a second while I work out what to tell you.",
  "busy",
);
