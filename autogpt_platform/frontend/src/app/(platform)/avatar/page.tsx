"use client";

import { Text } from "@/components/atoms/Text/Text";
import { BotAvatar } from "@/components/molecules/BotAvatar/BotAvatar";
import { AvatarEditor } from "@/components/organisms/AvatarEditor/AvatarEditor";
import { AccessoryStrip } from "./components/AccessoryStrip";
import { AvatarResultPanel } from "./components/AvatarResultPanel";
import { PreviewControls } from "./components/PreviewControls";
import { useAvatarPage } from "./useAvatarPage";

const MAIN_CLASS =
  "mx-auto w-full max-w-[1180px] space-y-6 px-4 pb-16 pt-6 sm:px-8 md:px-12";

export default function AvatarPage() {
  const {
    config,
    setConfig,
    status,
    setStatus,
    expression,
    setExpression,
    previewSize,
    setPreviewSize,
    url,
    copyUrl,
    downloadSvg,
  } = useAvatarPage();

  return (
    <main className={MAIN_CLASS}>
      <header className="flex flex-col gap-1">
        <Text variant="h2" as="h1">
          Avatar maker
        </Text>
        <Text variant="body" tone="muted">
          Build a face, try every accessory, and take the URL away with you.
        </Text>
      </header>

      <section className="rounded-2xl border border-zinc-200 bg-white p-4 sm:p-6">
        <AvatarEditor
          value={config}
          onChange={setConfig}
          status={status}
          size={180}
        />
      </section>

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,320px)]">
        <PreviewControls
          status={status}
          onStatusChange={setStatus}
          expression={expression}
          onExpressionChange={setExpression}
          previewSize={previewSize}
          onPreviewSizeChange={setPreviewSize}
        />
        <div className="flex flex-col gap-6">
          <section className="flex flex-col items-center gap-3 rounded-2xl border border-zinc-200 bg-white p-4">
            <Text variant="small" tone="muted" as="h2">
              At {previewSize}px
            </Text>
            <BotAvatar
              key={`${status}-${expression}`}
              config={config}
              size={previewSize}
              status={status}
              expression={expression === "auto" ? undefined : expression}
              title="Sized preview"
            />
          </section>
          <AvatarResultPanel
            url={url}
            onCopy={copyUrl}
            onDownload={downloadSvg}
          />
        </div>
      </div>

      <AccessoryStrip config={config} onPick={setConfig} />
    </main>
  );
}
