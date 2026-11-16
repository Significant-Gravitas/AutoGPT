import type { Meta, StoryObj } from "@storybook/nextjs";
import { CallAura } from "./CallAura";
import { CallFinishing } from "./CallFinishing";

interface Args {
  headline: string | null;
  stage: number;
}

// The wait after a call as the step shows it: Otto working, what ended the
// call, and the stage its words are at.
function FinishingScreen({ headline, stage }: Args) {
  return (
    <div className="flex w-full max-w-sm flex-col items-center gap-5 px-4 py-10">
      <CallAura levels={null} isActive={false} isWorking />
      <CallFinishing headline={headline} stage={stage} />
    </div>
  );
}

const meta: Meta<typeof FinishingScreen> = {
  title: "Onboarding/Interview/Call finishing",
  component: FinishingScreen,
  args: { headline: "That's everything Otto needed.", stage: 0 },
  argTypes: {
    stage: { control: { type: "inline-radio" }, options: [0, 1, 2] },
  },
};

export default meta;
type Story = StoryObj<typeof FinishingScreen>;

export const SavingWhatYouSaid: Story = {};

export const ReadingItBack: Story = { args: { stage: 1 } };

export const BuildingYourPlan: Story = { args: { stage: 2 } };

export const LostCall: Story = {
  args: { headline: "We lost the call.", stage: 0 },
};
