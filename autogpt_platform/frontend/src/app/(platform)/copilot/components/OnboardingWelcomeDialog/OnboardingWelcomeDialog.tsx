"use client";

import { usePostV1CompleteOnboardingStep } from "@/app/api/__generated__/endpoints/onboarding/onboarding";
import { Button } from "@/components/atoms/Button/Button";
import { GlassPixelBackdrop } from "@/components/atoms/GlassPixelBackdrop/GlassPixelBackdrop";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { trackBrainDump } from "@/services/onboarding/brain-dump-analytics";
import {
  ArrowLeft01Icon,
  BrainIcon,
  ElectricPlugsIcon,
  GraduationCapIcon,
  SparklesIcon,
  Sun01Icon,
  UserGroupIcon,
} from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";
import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useRef, useState } from "react";
import { useMeasuredHeight } from "@/components/contextual/IntegrationsPanel/components/ConnectServiceDialog/useMeasuredHeight";
import { ConnectToolsPanel } from "./ConnectToolsPanel";
import { isKey } from "@/lib/keyboard";

interface Props {
  isOpen: boolean;
  onClose: () => void;
}

interface CapabilityCard {
  title: string;
  // Capped at 20 words — an icon, short title, one sentence.
  body: string;
  /** Stage art: a zinc icon in a white tile. */
  icon?: IconSvgElement;
  /** Expands the dialog into the embedded provider picker. */
  cta?: { label: string };
}

const CARDS: CapabilityCard[] = [
  {
    title: "Meet AutoPilot.",
    body: "It does the work. Ask once, or put it on a schedule. It delivers while you do something else.",
    icon: SparklesIcon,
  },
  {
    title: "It works inside your tools.",
    body: "Slack, Gmail, Notion, GitHub and 40+ more. 500+ blocks under the hood.",
    icon: ElectricPlugsIcon,
    cta: { label: "Connect your tools" },
  },
  {
    title: "It learns how you operate.",
    body: "Teach it Skills, like an employee handbook. It keeps its own files, so it never starts from scratch.",
    icon: GraduationCapIcon,
  },
  {
    title: "It remembers everything.",
    body: "Memory beyond any human brain. It even dreams. Manage it all in the Agents tab.",
    icon: BrainIcon,
  },
];

// With the team feature on the deck stops being about AutoPilot's own
// abilities: AutoPilot is the Head of AI, and what the user is about to
// meet on the greeting page is a team.
const TEAM_CARDS: CapabilityCard[] = [
  {
    title: "Meet your Head of AI.",
    body: "AutoPilot is yours alone — never shared. It listens, diagnoses, and builds the team that does the work.",
    icon: SparklesIcon,
  },
  {
    title: "Hire an expert, or raise your own.",
    body: "Pick a ready-made expert for marketing, sales or ops, or describe a role and raise one from scratch.",
    icon: UserGroupIcon,
  },
  {
    title: "It works inside your tools.",
    body: "Slack, Gmail, Notion, GitHub and 40+ more. 500+ blocks under the hood.",
    icon: ElectricPlugsIcon,
    cta: { label: "Connect your tools" },
  },
  {
    title: "Every morning, a briefing.",
    body: "What the team did, what needs you, what's next — before you open a tab.",
    icon: Sun01Icon,
  },
];

// First-run capability cards on the copilot home (ChatGPT/Claude style):
// a stage on top carrying the card's icon, copy below,
// skippable at any card, shown once — completion is recorded server-side
// as the CAPABILITY_CARDS onboarding step. It also buys the background
// pipeline its last seconds: the greeting is only fetched, and only
// starts animating, after this closes.
export function OnboardingWelcomeDialog({ isOpen, onClose }: Props) {
  const [cardIndex, setCardIndex] = useState(0);
  // The CTA resizes the dialog into the provider picker instead of
  // navigating away — connecting must never close this dialog.
  const [isConnectOpen, setIsConnectOpen] = useState(false);
  const [contentRef, contentHeight] = useMeasuredHeight<HTMLDivElement>();
  const dialogRef = useRef<HTMLDivElement>(null);
  const { toast } = useToast();
  const { mutate: completeStep } = usePostV1CompleteOnboardingStep({
    mutation: {
      onError: () =>
        toast({
          title: "Could not save your onboarding progress",
          description: "You may see this introduction again next time.",
          variant: "destructive",
        }),
    },
  });
  // Child of HIRE_EXPERTS: without hiring there is no team to promise.
  const isExpertTeamFlagOn = useGetFlag(Flag.ONBOARDING_EXPERT_TEAM);
  const isHireExpertsFlagOn = useGetFlag(Flag.HIRE_EXPERTS);
  const isTeamEnabled = Boolean(isExpertTeamFlagOn && isHireExpertsFlagOn);
  const deck = isTeamEnabled ? "team" : "autopilot";
  const cards = isTeamEnabled ? TEAM_CARDS : CARDS;
  const card = cards[cardIndex];
  const isLastCard = cardIndex === cards.length - 1;

  function finish(outcome: "completed" | "skipped") {
    trackBrainDump(
      outcome === "completed"
        ? "capability_cards_completed"
        : "capability_cards_skipped",
      { card_index: cardIndex, deck },
    );
    completeStep({ params: { step: "CAPABILITY_CARDS" } });
    onClose();
  }

  // Escape ends the cards, and focus starts inside the card rather than on
  // whatever was behind the overlay. While the provider picker is open
  // Escape belongs to it — stepping back out of a half-typed API key must
  // not end onboarding for good.
  useEffect(() => {
    if (!isOpen) return;
    dialogRef.current?.focus();
    function handleKeyDown(event: KeyboardEvent) {
      if (isKey(event, "Escape") && !isConnectOpen) finish("skipped");
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isOpen, cardIndex, isConnectOpen, deck]);

  function handleNext() {
    if (isLastCard) {
      finish("completed");
      return;
    }
    trackBrainDump("capability_card_viewed", {
      card_index: cardIndex + 1,
      deck,
    });
    setCardIndex(cardIndex + 1);
  }

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.4, ease: [0.4, 0, 0.2, 1] }}
          className="fixed inset-0 z-[100] flex items-center justify-center bg-white/30 px-4 backdrop-blur-sm"
          data-testid="onboarding-welcome-overlay"
          role="dialog"
          aria-modal="true"
          aria-label="Welcome to AutoPilot"
        >
          <motion.div
            initial={{ opacity: 0, y: 16, scale: 0.97, maxWidth: "20rem" }}
            animate={{
              opacity: 1,
              y: 0,
              scale: 1,
              maxWidth: isConnectOpen ? "30rem" : "20rem",
            }}
            transition={{ duration: 0.45, ease: [0, 0, 0.2, 1] }}
            className="w-full max-w-[20rem] overflow-hidden rounded-xl border border-zinc-200 bg-white shadow-lg outline-none"
            ref={dialogRef}
            tabIndex={-1}
          >
            {/* Card-resize morph: the dialog animates between the compact
                capability card and the wider provider picker; height tracks
                whichever view is mounted. */}
            <motion.div
              animate={{ height: contentHeight ?? "auto" }}
              transition={{ duration: 0.35, ease: [0, 0, 0.2, 1] }}
            >
              <div ref={contentRef}>
                <AnimatePresence mode="wait" initial={false}>
                  {isConnectOpen ? (
                    <motion.div
                      key="connect"
                      initial={{ opacity: 0, y: 8 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -8 }}
                      transition={{ duration: 0.2, ease: [0, 0, 0.2, 1] }}
                    >
                      <ConnectToolsPanel
                        onBack={() => setIsConnectOpen(false)}
                        onNext={() => {
                          setIsConnectOpen(false);
                          handleNext();
                        }}
                      />
                    </motion.div>
                  ) : (
                    <motion.div
                      key="cards"
                      initial={{ opacity: 0, y: 8 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -8 }}
                      transition={{ duration: 0.2, ease: [0, 0, 0.2, 1] }}
                    >
                      {/* Stage: the card's icon floats here. */}
                      <div className="relative h-36 bg-gradient-to-br from-violet-100 via-violet-200 to-violet-300">
                        <GlassPixelBackdrop />
                        {cardIndex > 0 && (
                          <button
                            type="button"
                            aria-label="Previous card"
                            onClick={() => setCardIndex(cardIndex - 1)}
                            className="absolute left-3 top-3 z-10 flex size-5 items-center justify-center rounded-full text-violet-800/70 transition-colors hover:bg-white/50"
                          >
                            <Icon icon={ArrowLeft01Icon} size={13} />
                          </button>
                        )}
                        <AnimatePresence mode="wait">
                          <motion.div
                            key={cardIndex}
                            initial={{ opacity: 0, y: 12 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -12 }}
                            transition={{ duration: 0.3, ease: [0, 0, 0.2, 1] }}
                            className="absolute inset-0"
                            data-testid="capability-card"
                          >
                            {card.icon && (
                              <div className="flex h-full items-center justify-center">
                                <div className="flex size-14 items-center justify-center rounded-2xl bg-white shadow-md">
                                  <Icon
                                    icon={card.icon}
                                    size={28}
                                    className="text-violet-600"
                                  />
                                </div>
                              </div>
                            )}
                          </motion.div>
                        </AnimatePresence>
                      </div>

                      {/* Copy + controls, reference-style white lower half. */}
                      <div className="flex flex-col gap-2 px-5 pb-5 pt-4 text-left">
                        <Text variant="h5" tone="primary">
                          {card.title}
                        </Text>
                        <Text variant="body" tone="secondary">
                          {card.body}
                        </Text>
                        {card.cta && (
                          <Button
                            variant="ghost"
                            size="xs"
                            className="-ml-2.5 w-fit"
                            onClick={() => setIsConnectOpen(true)}
                          >
                            {card.cta.label}
                          </Button>
                        )}

                        <div className="mt-3 flex items-center justify-between">
                          <div className="flex items-center gap-1.5">
                            {cards.map((_, index) => (
                              <span
                                key={index}
                                className={
                                  index === cardIndex
                                    ? "h-1.5 w-4 rounded-full bg-zinc-900 transition-all"
                                    : "size-1.5 rounded-full bg-zinc-300 transition-all"
                                }
                              />
                            ))}
                          </div>
                          <div className="flex items-center gap-2">
                            <Button
                              variant="ghost"
                              size="xs"
                              onClick={() => finish("skipped")}
                            >
                              Skip
                            </Button>
                            <Button
                              variant="primary"
                              size="xs"
                              onClick={handleNext}
                            >
                              {!isLastCard
                                ? "Next"
                                : isTeamEnabled
                                  ? "Meet your team"
                                  : "Meet AutoPilot"}
                            </Button>
                          </div>
                        </div>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            </motion.div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
