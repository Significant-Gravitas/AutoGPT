"use client";

export default function ElevenLabsWidget() {
  return (
    <>
      {/* @ts-expect-error - this is a custom element */}
      <elevenlabs-convai agent-id="agent_01k0catqvjef0sk50r03cj49ek"></elevenlabs-convai>
      <script
        src="https://unpkg.com/@elevenlabs/convai-widget-embed"
        async
        type="text/javascript"
      ></script>
    </>
  );
}
