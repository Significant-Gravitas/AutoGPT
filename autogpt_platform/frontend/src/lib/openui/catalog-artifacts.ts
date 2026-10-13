import { defineComponent } from "@openuidev/lang-core";
import { z } from "zod/v4";
import { shortText } from "./data-values";

export const artifactURI = z
  .string()
  .max(4000)
  .refine((value) => {
    if (
      /^workspace:\/\/[a-zA-Z0-9_-]+(?:#[a-zA-Z0-9.+-]+\/[a-zA-Z0-9.+-]+)?$/.test(
        value,
      )
    )
      return true;
    try {
      const url = new URL(value);
      return (
        ["https:", "http:"].includes(url.protocol) &&
        !url.username &&
        !url.password
      );
    } catch {
      return false;
    }
  }, "Use an actual workspace file URI or HTTP(S) output URL; never a guessed path, data URI or executable URL.");

export const Artifact = defineComponent({
  name: "Artifact",
  description:
    "An inline card for an actual generated or uploaded artifact. Reuses native image/video/audio renderers and file downloads. uri must be the exact tool-returned workspace://file-id#mime/type or HTTP(S) URL, not invented. Provide filename, accurate mimeType, description (also image alt text). No script or HTML execution. For multiple generated images use multiple cards; for choosing images use ResultList with real image URLs. For editable text use DocumentEditor inside Form, preserving its source text. Do not claim an image was generated until the generation tool returned it.",
  props: z.object({
    title: shortText,
    uri: artifactURI,
    mimeType: z.string().regex(/^[a-zA-Z0-9.+-]+\/[a-zA-Z0-9.+-]+$/),
    filename: shortText,
    description: z.string().max(2000),
  }),
  component: null,
});

export const ExecutionView = defineComponent({
  name: "ExecutionView",
  description:
    "Inline live status and outputs of an actual AutoGPT agent execution. Use only tool-returned graphId and executionId. Status, outputs and cancel come from the authenticated backend, never model-written progress. Failed runs offer a retry request in chat. This component cannot start work or approve external writes. Keep native approval cards for approval.",
  props: z.object({
    title: shortText,
    graphId: z.string().uuid(),
    executionId: z.string().uuid(),
  }),
  component: null,
});
