"use client";

import { usePostV1CompleteOnboardingStep } from "@/app/api/__generated__/endpoints/onboarding/onboarding";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { trackBrainDump } from "@/services/onboarding/brain-dump-analytics";
import {
  ArrowLeft01Icon,
  BrainIcon,
  ElectricPlugsIcon,
  GraduationCapIcon,
  SparklesIcon,
} from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";
import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useRef, useState } from "react";
import { useMeasuredHeight } from "@/components/contextual/IntegrationsPanel/components/ConnectServiceDialog/useMeasuredHeight";
import { ConnectToolsPanel } from "./ConnectToolsPanel";
import { GlassPixelBackdrop } from "@/components/atoms/GlassPixelBackdrop/GlassPixelBackdrop";

interface Props {
  isOpen: boolean;
  onClose: () => void;
}

interface CapabilityCard {
  title: string;
  // Capped at 20 words — an icon, short title, one sentence.
  body: string;
  /** Stage art: big duotone-violet icon in a white tile. */
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

// First-run capability cards on the copilot home (ChatGPT/Claude style):
// a tinted stage on top carrying the card's icon, copy below,
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
  const card = CARDS[cardIndex];
  const isLastCard = cardIndex === CARDS.length - 1;

  function finish(outcome: "completed" | "skipped") {
    trackBrainDump(
      outcome === "completed"
        ? "capability_cards_completed"
        : "capability_cards_skipped",
      { card_index: cardIndex },
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
      if (event.key === "Escape" && !isConnectOpen) finish("skipped");
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isOpen, cardIndex, isConnectOpen]);

  function handleNext() {
    if (isLastCard) {
      finish("completed");
      return;
    }
    trackBrainDump("capability_card_viewed", { card_index: cardIndex + 1 });
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
            initial={{ opacity: 0, y: 16, scale: 0.97, maxWidth: "26rem" }}
            animate={{
              opacity: 1,
              y: 0,
              scale: 1,
              maxWidth: isConnectOpen ? "30rem" : "26rem",
            }}
            transition={{ duration: 0.45, ease: [0, 0, 0.2, 1] }}
            className="w-full max-w-[26rem] overflow-hidden rounded-3xl bg-white shadow-[0_24px_80px_-24px_rgba(0,0,0,0.3)] outline-none"
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
                      {/* Tinted stage: the card's icon floats here. */}
                      <div className="relative h-44 bg-gradient-to-br from-[#e6dbff] via-[#ddccff] to-[#d0b9ff]">
                        <GlassPixelBackdrop />
                        <span className="absolute left-5 top-4 z-10 flex items-center gap-1 text-xs font-medium text-[#5b21b6]/70">
                          {cardIndex > 0 && (
                            <button
                              type="button"
                              aria-label="Previous card"
                              onClick={() => setCardIndex(cardIndex - 1)}
                              className="-ml-1 flex h-5 w-5 items-center justify-center rounded-full transition-colors hover:bg-white/50"
                            >
                              <Icon icon={ArrowLeft01Icon} size={13} />
                            </button>
                          )}
                          {cardIndex + 1} of {CARDS.length}
                        </span>
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
                                <div className="flex h-20 w-20 items-center justify-center rounded-3xl bg-white shadow-lg">
                                  <Icon
                                    icon={card.icon}
                                    size={40}
                                    className="text-violet-600"
                                  />
                                </div>
                              </div>
                            )}
                          </motion.div>
                        </AnimatePresence>
                      </div>

                      {/* Copy + controls, reference-style white lower half. */}
                      <div className="flex flex-col gap-3 px-7 pb-7 pt-6 text-left">
                        <Text
                          variant="h3"
                          className="!text-[1.25rem] text-zinc-900"
                        >
                          {card.title}
                        </Text>
                        <Text
                          variant="body"
                          className="!text-[0.9375rem] !text-zinc-600"
                        >
                          {card.body}
                        </Text>
                        {card.cta && (
                          <button
                            type="button"
                            onClick={() => setIsConnectOpen(true)}
                            className="w-fit text-sm font-medium text-violet-600 underline-offset-4 hover:underline"
                          >
                            {card.cta.label}
                          </button>
                        )}

                        <div className="mt-3 flex items-center justify-between">
                          <div className="flex items-center gap-2">
                            {CARDS.map((_, index) => (
                              <span
                                key={index}
                                className={
                                  index === cardIndex
                                    ? "h-2 w-6 rounded-full bg-violet-500 transition-all"
                                    : "h-2 w-2 rounded-full bg-zinc-200 transition-all"
                                }
                              />
                            ))}
                          </div>
                          <div className="flex items-center gap-3">
                            <Button
                              variant="secondary"
                              size="small"
                              onClick={() => finish("skipped")}
                            >
                              Skip
                            </Button>
                            <Button
                              variant="primary"
                              size="small"
                              onClick={handleNext}
                            >
                              {isLastCard ? "Meet AutoPilot" : "Next"}
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
