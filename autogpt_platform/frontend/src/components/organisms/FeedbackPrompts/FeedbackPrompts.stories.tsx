import { Text } from "@/components/atoms/Text/Text";
import { AutopilotAvatar } from "@/components/molecules/AutopilotAvatar/AutopilotAvatar";
import { AUTOPILOT_AVATAR_URL } from "@/components/molecules/AutopilotAvatar/helpers";
import type { Meta, StoryObj } from "@storybook/nextjs";
import { CancelTrialFeedbackDialog } from "./CancelTrialFeedbackDialog";
import { CheckoutAbandonedDialog } from "./CheckoutAbandonedDialog";
import { NpsCard } from "./NpsCard";
import { OnboardingSkipPrompt } from "./OnboardingSkipPrompt";
import { OttoMistakeCard } from "./OttoMistakeCard";
import { TopUpDismissedCard } from "./TopUpDismissedCard";
import { WorkCheckCard } from "./WorkCheckCard";
import { MockPage, SkeletonBlocks } from "./components/MockPage";
import {
  BRAIN_DUMP_SKIP_REASONS,
  HIRE_SKIP_REASONS,
  SAMPLE_STARTERS,
} from "./helpers";

const meta: Meta = {
  title: "Prototypes/Feedback prompts",
  parameters: { layout: "fullscreen" },
};

export default meta;
type Story = StoryObj;

const MARIA_AVATAR = "/autogpt-characters/v1.1/expert-maria/neutral/128.webp";

export const TrialCancelReason: Story = {
  name: "1a. Trial cancel: reason",
  render: () => (
    <MockPage title="Billing">
      <SkeletonBlocks />
      <CancelTrialFeedbackDialog />
    </MockPage>
  ),
};

function TrialOffer({ starters }: { starters: string[] }) {
  return (
    <MockPage title="Billing">
      <SkeletonBlocks />
      <CancelTrialFeedbackDialog
        starters={starters}
        initialStep="offer"
        initialReason="I haven't seen results yet"
      />
    </MockPage>
  );
}

export const TrialCancelOfferDeveloper: Story = {
  name: "1b. Trial offer: user said they're a developer",
  render: () => <TrialOffer starters={SAMPLE_STARTERS.developer} />,
};

export const TrialCancelOfferSales: Story = {
  name: "1c. Trial offer: user said they're in sales",
  render: () => <TrialOffer starters={SAMPLE_STARTERS.sales} />,
};

export const TrialCancelOfferNoAnswers: Story = {
  name: "1d. Trial offer: no onboarding answers stored",
  render: () => <TrialOffer starters={[]} />,
};

export const CheckoutAbandoned: Story = {
  name: "2. Back from checkout without paying",
  render: () => (
    <MockPage title="Billing">
      <SkeletonBlocks />
      <CheckoutAbandonedDialog initialReason="My payment failed" />
    </MockPage>
  ),
};

function FinishedTask({ title, output }: { title: string; output: string }) {
  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-center gap-3">
        <Text variant="h4">{title}</Text>
        <span className="rounded-full bg-success-surface px-2.5 py-0.5 text-caption font-medium text-success-foreground">
          Done
        </span>
      </div>
      <div className="flex flex-col gap-2 rounded-2xl border border-border-base bg-surface p-5">
        <Text variant="small" className="!text-foreground-tertiary">
          Result
        </Text>
        <Text variant="body">{output}</Text>
      </div>
      <div className="h-40 rounded-2xl border border-border-base bg-surface" />
    </div>
  );
}

export const WorkCheckFirstOtto: Story = {
  name: "3a. Work check: Otto's first finished task",
  render: () => (
    <MockPage
      title="Tasks"
      corner={
        <WorkCheckCard
          name="Otto"
          avatarUrl={AUTOPILOT_AVATAR_URL}
          task="Research 10 leads"
          eyebrow="Otto's first finished task"
        />
      }
    >
      <FinishedTask
        title="Research 10 leads"
        output="Found 10 Austin dental practices with 3+ locations, each with the owner's name and a public email from their site."
      />
    </MockPage>
  ),
};

export const WorkCheckWeeklyExpert: Story = {
  name: "3b. Work check: weekly spot check on an expert",
  render: () => (
    <MockPage
      title="Tasks"
      corner={
        <WorkCheckCard
          name="Maria"
          avatarUrl={MARIA_AVATAR}
          task="Rewrite the pricing page"
          eyebrow="Quick check-in"
        />
      }
    >
      <FinishedTask
        title="Rewrite the pricing page"
        output="New draft is in your Drive: a shorter hero, plans compared in one table, and a FAQ for the five questions sales hears most."
      />
    </MockPage>
  ),
};

export const WorkCheckNotGood: Story = {
  name: "3c. Work check: after “Not good”",
  render: () => (
    <MockPage
      title="Tasks"
      corner={
        <WorkCheckCard
          name="Maria"
          avatarUrl={MARIA_AVATAR}
          task="Rewrite the pricing page"
          eyebrow="Quick check-in"
          initialRating="bad"
        />
      }
    >
      <FinishedTask
        title="Rewrite the pricing page"
        output="New draft is in your Drive: a shorter hero, plans compared in one table, and a FAQ for the five questions sales hears most."
      />
    </MockPage>
  ),
};

function UserBubble({ children }: { children: string }) {
  return (
    <div className="ml-auto max-w-md rounded-2xl bg-surface-active px-4 py-3">
      <Text variant="body">{children}</Text>
    </div>
  );
}

function OttoTurn({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex max-w-2xl gap-3">
      <AutopilotAvatar size={28} className="ring-1 ring-border-base" />
      <div className="flex flex-1 flex-col gap-3">{children}</div>
    </div>
  );
}

export const OttoMistake: Story = {
  name: "4. Otto senses it got something wrong",
  render: () => (
    <MockPage title="Otto">
      <div className="flex flex-col gap-6">
        <UserBubble>
          Find me 20 dentists in Austin with their emails.
        </UserBubble>
        <OttoTurn>
          <Text variant="body">
            Here are 20 practices with contact emails. I&apos;ve saved them to a
            sheet called “Austin dentists”.
          </Text>
        </OttoTurn>
        <UserBubble>
          This is useless. Half of these are in Dallas and the emails bounce.
          Did you just make them up??
        </UserBubble>
        <OttoTurn>
          <Text variant="body">
            You&apos;re right, and I&apos;m sorry. I let Dallas results into the
            search and filled gaps with guessed emails instead of leaving them
            blank. I&apos;m redoing it now with Austin only, and only emails
            published on each practice&apos;s site.
          </Text>
          <OttoMistakeCard />
        </OttoTurn>
      </div>
    </MockPage>
  ),
};

export const TopUpDismissed: Story = {
  name: "5. Closed the top-up prompt without buying",
  render: () => <MockPage title="Home" corner={<TopUpDismissedCard />} />,
};

function OnboardingStep({
  heading,
  body,
  question,
  reasons,
}: {
  heading: string;
  body: string;
  question: string;
  reasons: typeof HIRE_SKIP_REASONS;
}) {
  return (
    <MockPage title="" centered>
      <Text variant="h3" className="text-center">
        {heading}
      </Text>
      <Text variant="body" className="max-w-md text-center">
        {body}
      </Text>
      <div className="size-40 rounded-full bg-surface-active" />
      <Text variant="small" className="!text-foreground-tertiary">
        Step skipped
      </Text>
      <OnboardingSkipPrompt question={question} reasons={reasons} />
    </MockPage>
  );
}

export const BrainDumpSkipped: Story = {
  name: "6. Onboarding: skipped the brain dump",
  render: () => (
    <OnboardingStep
      heading="Tell Otto about your work"
      body="Talk for a minute about what fills your week. Otto uses it to suggest where to start."
      question="No problem. Mind telling us why?"
      reasons={BRAIN_DUMP_SKIP_REASONS}
    />
  ),
};

export const HireSkipped: Story = {
  name: "7. Onboarding: left the hire step without picking",
  render: () => (
    <OnboardingStep
      heading="Pick your first experts"
      body="Each expert takes one kind of work off your plate."
      question="What would have helped you pick?"
      reasons={HIRE_SKIP_REASONS}
    />
  ),
};

export const NpsAsk: Story = {
  name: "8a. Day 14: NPS",
  render: () => <MockPage title="Home" corner={<NpsCard />} />,
};

export const NpsScored: Story = {
  name: "8b. Day 14: NPS after a score",
  render: () => <MockPage title="Home" corner={<NpsCard initialScore={6} />} />,
};
