"use client";

import { Button } from "@/components/atoms/Button/Button";
import { cn } from "@/lib/utils";
import { Icon } from "@/components/atoms/Icon/Icon";
import { VoicePicker } from "@/components/organisms/VoicePicker/VoicePicker";
import {
  ArrowLeft02Icon,
  ArrowRight02Icon,
  RefreshIcon,
} from "@hugeicons/core-free-icons";
import {
  BEAT_KEYS,
  questionId,
  type BeatKey,
  type RaiseFlowItem,
} from "../../flowItems";
import { VOICE_SAMPLES } from "../../helpers";
import { useRef, useState } from "react";
import { useRaisePage } from "../../useRaisePage";
import { useConversationScroll } from "../../useConversationScroll";
import { AboutStep } from "../AboutStep/AboutStep";
import { AutoGPTBubble } from "../AutoGPTBubble/AutoGPTBubble";
import { AvatarStep } from "../AvatarStep/AvatarStep";
import { ColorStep } from "../ColorStep/ColorStep";
import {
  ditherColorsFor,
  interactiveCardClassFor,
  selectedCardClassFor,
  textClassFor,
} from "../ColorStep/helpers";
import { DitheredWaves } from "../DitheredWaves/DitheredWaves";
import { BudgetStep } from "../BudgetStep/BudgetStep";
import { MarketplaceStep } from "../MarketplaceStep/MarketplaceStep";
import { NameStep } from "../NameStep/NameStep";
import { RoleStep } from "../RoleStep/RoleStep";
import { SkillsStep } from "../SkillsStep/SkillsStep";
import { nameSuggestionsFor } from "../RoleStep/helpers";
import { SoulPreviewPanel } from "../SoulPreviewPanel/SoulPreviewPanel";
import { RestartConfirmDialog } from "./components/RestartConfirmDialog";

const STEP_ANIMATION =
  "duration-500 animate-in fade-in slide-in-from-bottom-2 fill-mode-both motion-reduce:animate-none";

export function RaiseFlow() {
  const {
    hasStarted,
    items,
    role,
    name,
    color,
    avatarUrl,
    about,
    voiceLabel,
    budget,
    marketplace,
    skills,
    kit,
    isSubmitting,
    canGoBack,
    startRaising,
    restart,
    goBack,
    revealStep,
    pickRole,
    submitName,
    pickColor,
    pickAvatar,
    skipAvatar,
    submitAbout,
    skipAbout,
    pickVoice,
    skipVoice,
    submitBudget,
    skipBudget,
    submitMarketplace,
    skipMarketplace,
    submitSkills,
    skipSkills,
  } = useRaisePage();
  const { scrollRef, canScrollUp, canScrollDown } = useConversationScroll();
  const [isRestartOpen, setIsRestartOpen] = useState(false);
  const initialMessageIds = useRef<Set<string> | null>(null);
  if (initialMessageIds.current === null) {
    initialMessageIds.current = new Set(
      items.filter((item) => item.kind === "message").map((item) => item.id),
    );
  }
  const seenMessageIds = initialMessageIds.current;

  function renderStep(beat: BeatKey) {
    switch (beat) {
      case "role":
        return <RoleStep selectedRole={role} color={color} onPick={pickRole} />;
      case "name":
        return (
          <NameStep
            selectedName={name || null}
            suggestions={nameSuggestionsFor(role)}
            color={color}
            onSubmit={submitName}
          />
        );
      case "color":
        return <ColorStep selectedColor={color} onPick={pickColor} />;
      case "avatar":
        return (
          <AvatarStep
            name={name}
            color={color}
            avatarUrl={avatarUrl || null}
            isSkipped={avatarUrl === ""}
            onPick={pickAvatar}
            onSkip={skipAvatar}
          />
        );
      case "about":
        return (
          <AboutStep
            submittedAbout={about}
            name={name}
            color={color}
            onSubmit={submitAbout}
            onSkip={skipAbout}
          />
        );
      case "voice":
        return (
          <VoicePicker
            name={name}
            samples={VOICE_SAMPLES}
            hideHeader
            labelClassName={textClassFor(color)}
            cardColors={{
              selected: selectedCardClassFor(color),
              interactive: interactiveCardClassFor(color),
            }}
            onPick={pickVoice}
            onSkip={skipVoice}
          />
        );
      case "budget":
        return (
          <BudgetStep
            color={color}
            submittedBudget={budget}
            onSubmit={submitBudget}
            onSkip={skipBudget}
          />
        );
      case "marketplace":
        return (
          <MarketplaceStep
            color={color}
            submitted={marketplace}
            onSubmit={submitMarketplace}
            onSkip={skipMarketplace}
          />
        );
      case "skills":
        return (
          <SkillsStep
            name={name}
            color={color}
            submitted={skills}
            existingCount={marketplace?.length ?? 0}
            isSubmitting={isSubmitting}
            onSubmit={submitSkills}
            onSkip={skipSkills}
          />
        );
    }
  }

  function renderItem(item: RaiseFlowItem) {
    if (item.kind === "startButton") {
      return (
        <div key={item.id} className={`flex delay-700 ${STEP_ANIMATION}`}>
          <Button
            variant="secondary"
            size="small"
            className="-ml-3 rounded-full"
            onClick={startRaising}
            disabled={hasStarted}
          >
            {hasStarted ? "Setting up expert now" : "Start raising"}
            {hasStarted ? null : <Icon icon={ArrowRight02Icon} size={14} />}
          </Button>
        </div>
      );
    }
    if (item.kind === "step") {
      return (
        <div key={item.id} id={item.id} className={STEP_ANIMATION}>
          {renderStep(item.beat)}
        </div>
      );
    }
    return (
      <AutoGPTBubble
        key={item.id}
        id={item.id}
        text={item.text}
        animate={!seenMessageIds.has(item.id)}
        onTypingComplete={revealOnTyped(item.id, revealStep)}
      />
    );
  }

  return (
    <main className="min-h-screen bg-background lg:h-screen lg:overflow-hidden">
      <div className="grid w-full items-stretch lg:h-full lg:grid-cols-2">
        <div className="relative order-2 lg:order-1 lg:h-screen">
          <div
            ref={scrollRef}
            role="log"
            aria-live="polite"
            aria-relevant="additions text"
            aria-label="Raise expert conversation"
            className="flex flex-col gap-4 px-4 pb-16 pt-6 scrollbar-none sm:px-6 lg:h-full lg:overflow-y-auto lg:px-8"
          >
            {items.map(renderItem)}
          </div>
          <ScrollFade edge="top" isVisible={canScrollUp} />
          <ScrollFade edge="bottom" isVisible={canScrollDown} />
        </div>

        <div className="relative order-1 m-2 overflow-hidden rounded-[2.5rem] bg-muted/40 lg:order-2">
          <DitheredWaves
            className="absolute inset-0"
            colors={ditherColorsFor(color)}
          />
          <div className="absolute left-4 top-4 z-10 flex gap-2 sm:left-6 sm:top-6">
            <Button
              variant="icon"
              size="small"
              aria-label="Back"
              className="bg-white p-2 hover:bg-zinc-100"
              onClick={goBack}
              disabled={!canGoBack}
            >
              <Icon icon={ArrowLeft02Icon} size={16} />
            </Button>
            <Button
              variant="icon"
              size="small"
              aria-label="Start over"
              className="bg-white p-2 hover:bg-zinc-100"
              onClick={() => setIsRestartOpen(true)}
            >
              <Icon icon={RefreshIcon} size={16} />
            </Button>
          </div>

          <div className="relative flex h-full items-center justify-center p-4 sm:p-6">
            <SoulPreviewPanel
              name={name}
              role={role}
              avatarUrl={avatarUrl || null}
              color={color}
              about={about}
              voiceLabel={voiceLabel}
              kit={kit}
            />
          </div>
        </div>
      </div>
      <RestartConfirmDialog
        open={isRestartOpen}
        onOpenChange={setIsRestartOpen}
        onConfirm={() => {
          setIsRestartOpen(false);
          restart();
        }}
      />
    </main>
  );
}

// Each question hands off to the controls it introduces once it has finished
// typing, so nothing appears mid-sentence.
function revealOnTyped(id: string, reveal: (beat: BeatKey) => void) {
  const beat = BEAT_KEYS.find((key) => questionId(key) === id);
  return beat ? () => reveal(beat) : undefined;
}

// Signals there is more conversation past the edge, fading into the page
// background so it reads as depth rather than a border.
function ScrollFade({
  edge,
  isVisible,
}: {
  edge: "top" | "bottom";
  isVisible: boolean;
}) {
  return (
    <div
      aria-hidden
      className={cn(
        "pointer-events-none absolute inset-x-0 h-16 transition-opacity duration-200",
        edge === "top"
          ? "top-0 bg-gradient-to-b from-background to-transparent"
          : "bottom-0 bg-gradient-to-t from-background to-transparent",
        isVisible ? "opacity-100" : "opacity-0",
      )}
    />
  );
}
