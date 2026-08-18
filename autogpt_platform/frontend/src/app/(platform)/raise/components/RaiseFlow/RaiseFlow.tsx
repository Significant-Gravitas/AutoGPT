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
import { useRaisePage } from "../../useRaisePage";
import { useConversationScroll } from "../../useConversationScroll";
import { AboutStep } from "../AboutStep/AboutStep";
import { AutoGPTBubble } from "../AutoGPTBubble/AutoGPTBubble";
import { AvatarStep } from "../AvatarStep/AvatarStep";
import { ColorStep } from "../ColorStep/ColorStep";
import {
  ditherColorsFor,
  interactiveCardClassFor,
  rowSelectedClassFor,
  selectedCardClassFor,
  textClassFor,
} from "../ColorStep/helpers";
import { DitheredWaves } from "../DitheredWaves/DitheredWaves";
import { FirstTaskStep } from "../FirstTaskStep/FirstTaskStep";
import { NameStep } from "../NameStep/NameStep";
import { RoleStep } from "../RoleStep/RoleStep";
import { nameSuggestionsFor } from "../RoleStep/helpers";
import { SoulPreviewPanel } from "../SoulPreviewPanel/SoulPreviewPanel";

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
    firstTask,
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
    submitFirstTask,
    skipFirstTask,
  } = useRaisePage();
  const { scrollRef, canScrollUp, canScrollDown } = useConversationScroll();

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
            role={role}
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
              selectedRow: rowSelectedClassFor(color),
            }}
            onPick={pickVoice}
            onSkip={skipVoice}
          />
        );
      case "firstTask":
        return (
          <FirstTaskStep
            name={name}
            color={color}
            submittedTask={firstTask}
            isSubmitting={isSubmitting}
            onSubmit={submitFirstTask}
            onSkip={skipFirstTask}
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
        <div key={item.id} className={STEP_ANIMATION}>
          {renderStep(item.beat)}
        </div>
      );
    }
    return (
      <AutoGPTBubble
        key={item.id}
        text={item.text}
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
          <div className="absolute right-4 top-4 z-10 flex gap-2 sm:right-6 sm:top-6">
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
            {/* Testing only: wipes the persisted draft and replays the flow. */}
            <Button
              variant="icon"
              size="small"
              aria-label="Restart flow"
              className="bg-white p-2 hover:bg-zinc-100"
              onClick={restart}
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
              firstTask={firstTask}
            />
          </div>
        </div>
      </div>
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
