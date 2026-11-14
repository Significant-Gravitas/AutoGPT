import { Text } from "@/components/atoms/Text/Text";
import { AUTOPILOT_AVATAR_URL } from "@/components/molecules/AutopilotAvatar/helpers";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import type { Meta, StoryObj } from "@storybook/nextjs";
import { CancelTrialFeedbackDialog } from "./CancelTrialFeedbackDialog";
import { CheckoutAbandonedDialog } from "./CheckoutAbandonedDialog";
import { NpsCard } from "./NpsCard";
import { OttoMistakeCard } from "./OttoMistakeCard";
import { TOP_UP_QUESTION, TopUpQuestion } from "./TopUpQuestion";
import { WorkCheckCard } from "./WorkCheckCard";
import { MockPage, SkeletonBlocks } from "./components/MockPage";

const meta: Meta = {
  title: "Organisms/Feedback prompts",
  parameters: { layout: "fullscreen" },
};

export default meta;
type Story = StoryObj;

const MARIA_AVATAR = "/autogpt-characters/v1.1/expert-maria/neutral/128.webp";

export const TrialCancel: Story = {
  name: "1a. Trial cancel",
  render: () => (
    <MockPage title="Billing">
      <SkeletonBlocks />
      <CancelTrialFeedbackDialog />
    </MockPage>
  ),
};

export const TrialCancelWithNote: Story = {
  name: "1b. Trial cancel: “It's missing something” picked",
  render: () => (
    <MockPage title="Billing">
      <SkeletonBlocks />
      <CancelTrialFeedbackDialog initialReason="missing_something" />
    </MockPage>
  ),
};

export const CheckoutUpgrade: Story = {
  name: "2. Left upgrade checkout without paying",
  render: () => (
    <MockPage title="Billing">
      <SkeletonBlocks />
      <CheckoutAbandonedDialog onTryAnotherCard={() => {}} />
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

function ReplyTurn({
  name,
  avatarUrl,
  children,
}: {
  name: string;
  avatarUrl: string;
  children: React.ReactNode;
}) {
  return (
    <div className="flex max-w-2xl gap-3">
      <ExpertAvatar
        name={name}
        avatarUrl={avatarUrl}
        size={28}
        className="ring-1 ring-border-base"
      />
      <div className="flex flex-1 flex-col gap-3">{children}</div>
    </div>
  );
}

function OttoTurn({ children }: { children: React.ReactNode }) {
  return (
    <ReplyTurn name="Otto" avatarUrl={AUTOPILOT_AVATAR_URL}>
      {children}
    </ReplyTurn>
  );
}

function OttoLeadsChat() {
  return (
    <MockPage title="Otto">
      <div className="flex flex-col gap-6">
        <UserBubble>
          Find 10 dental practices in Austin with 3+ locations, with the
          owner&apos;s name and email.
        </UserBubble>
        <OttoTurn>
          <Text variant="body">
            Done. I found 10 practices and saved them to a sheet called “Austin
            dental groups”, each with the owner&apos;s name and the email listed
            on their site.
          </Text>
          <WorkCheckCard
            name="Otto"
            avatarUrl={AUTOPILOT_AVATAR_URL}
            note="Quick check-in on your first finished task"
          />
        </OttoTurn>
      </div>
    </MockPage>
  );
}

function MariaPricingChat({ rating }: { rating: "bad" | "okay" | null }) {
  return (
    <MockPage title="Maria">
      <div className="flex flex-col gap-6">
        <UserBubble>
          Rewrite our pricing page so it&apos;s shorter and answers what sales
          gets asked most.
        </UserBubble>
        <ReplyTurn name="Maria" avatarUrl={MARIA_AVATAR}>
          <Text variant="body">
            The new draft is in your Drive: a shorter hero, the plans compared
            in one table, and a FAQ for the five questions sales hears most.
          </Text>
          <WorkCheckCard
            name="Maria"
            avatarUrl={MARIA_AVATAR}
            note="Quick check-in on this task"
            initialRating={rating}
          />
        </ReplyTurn>
      </div>
    </MockPage>
  );
}

export const WorkCheckFirstOtto: Story = {
  name: "3a. Work check: Otto's first finished task",
  render: () => <OttoLeadsChat />,
};

export const WorkCheckWeeklyExpert: Story = {
  name: "3b. Work check: weekly check on an expert's task",
  render: () => <MariaPricingChat rating={null} />,
};

export const WorkCheckNotGood: Story = {
  name: "3c. Work check: after “Not good”",
  render: () => <MariaPricingChat rating="bad" />,
};

export const WorkCheckOkay: Story = {
  name: "3d. Work check: after “Okay”",
  render: () => <MariaPricingChat rating="okay" />,
};

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

export const TopUpNotNow: Story = {
  name: "5. Out of credits: after “Not now”",
  render: () => (
    <MockPage title="Home">
      <SkeletonBlocks />
      <Dialog
        title={TOP_UP_QUESTION}
        styling={{ maxWidth: "28rem" }}
        controlled={{ isOpen: true, set: () => {} }}
      >
        <Dialog.Content>
          <TopUpQuestion />
        </Dialog.Content>
      </Dialog>
    </MockPage>
  ),
};

export const NpsAsk: Story = {
  name: "6a. Day 14: NPS",
  render: () => <MockPage title="Home" corner={<NpsCard />} />,
};

export const NpsScored: Story = {
  name: "6b. Day 14: NPS after a score",
  render: () => <MockPage title="Home" corner={<NpsCard initialScore={6} />} />,
};
