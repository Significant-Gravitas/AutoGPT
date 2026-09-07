"use client";

import { Text } from "@/components/atoms/Text/Text";
import { BotAvatar } from "@/components/molecules/BotAvatar/BotAvatar";
import { AccessoryPicker } from "./components/AccessoryPicker";
import { ActionBar } from "./components/ActionBar";
import { AvatarStage } from "./components/AvatarStage";
import { ColorPicker } from "./components/ColorPicker";
import { RosterPreview } from "./components/RosterPreview";
import { ShapePicker } from "./components/ShapePicker";
import { useAvatarPage } from "./useAvatarPage";

const MAIN_CLASS =
  "mx-auto min-h-screen w-full max-w-[960px] space-y-8 px-4 pb-20 pt-8 sm:px-8 duration-500 animate-in fade-in slide-in-from-bottom-2 fill-mode-both motion-reduce:animate-none";

export default function AvatarPage() {
  const {
    config,
    status,
    setStatus,
    turn,
    setTurn,
    outline,
    setOutline,
    setShape,
    setColor,
    setAccessory,
    shuffle,
    reset,
    copyLink,
    downloadSvg,
    downloadPng,
    isExporting,
    exportRef,
  } = useAvatarPage();

  return (
    <main className={MAIN_CLASS}>
      <header className="space-y-1">
        <Text variant="h2" as="h1" className="text-zinc-900">
          Avatar lab
        </Text>
        <Text variant="body" as="p" className="text-zinc-500">
          Shape, colour, one accessory. Expression comes from what the expert is
          doing.
        </Text>
      </header>
      <AvatarStage
        config={config}
        status={status}
        turn={turn}
        outline={outline}
        onStatusChange={setStatus}
        onTurnChange={setTurn}
        onOutlineChange={setOutline}
      />
      <ActionBar
        onShuffle={shuffle}
        onReset={reset}
        onCopyLink={copyLink}
        onDownloadSvg={downloadSvg}
        onDownloadPng={downloadPng}
        isExporting={isExporting}
      />
      <ShapePicker config={config} outline={outline} onSelect={setShape} />
      <ColorPicker selected={config.color} onSelect={setColor} />
      <AccessoryPicker
        config={config}
        outline={outline}
        onSelect={setAccessory}
      />
      <RosterPreview config={config} status={status} outline={outline} />
      <div ref={exportRef} hidden aria-hidden>
        <BotAvatar
          config={config}
          status={status}
          animated={false}
          outline={outline}
        />
      </div>
    </main>
  );
}
