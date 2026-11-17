import structuredClone from "@ungap/structured-clone";
import {
  TextDecoderStream,
  TextEncoderStream,
} from "@stardazed/streams-text-encoding";

// The AI SDK's stream parser expects these web globals; Hermes ships neither.
const target = globalThis as Record<string, unknown>;
if (!target.structuredClone) target.structuredClone = structuredClone;
if (!target.TextEncoderStream) target.TextEncoderStream = TextEncoderStream;
if (!target.TextDecoderStream) target.TextDecoderStream = TextDecoderStream;
