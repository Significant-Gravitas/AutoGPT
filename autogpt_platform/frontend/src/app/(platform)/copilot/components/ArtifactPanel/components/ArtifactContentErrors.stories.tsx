import { Image01Icon, Video01Icon } from "@hugeicons/core-free-icons";
import type { Meta, StoryObj } from "@storybook/nextjs";
import { http, HttpResponse } from "msw";
import type { ArtifactRef } from "../../../store";
import type { ArtifactClassification } from "../helpers";
import { ArtifactContent } from "./ArtifactContent";

const PROXY_BASE = "/api/proxy/api/workspace/files";

const IMAGE_ARTIFACT: ArtifactRef = {
  id: "broken-image",
  title: "chart.png",
  mimeType: "image/png",
  sourceUrl: `${PROXY_BASE}/broken-image/download`,
  origin: "agent",
};

const VIDEO_ARTIFACT: ArtifactRef = {
  id: "broken-video",
  title: "walkthrough.mp4",
  mimeType: "video/mp4",
  sourceUrl: `${PROXY_BASE}/broken-video/download`,
  origin: "agent",
};

const IMAGE_CLASSIFICATION: ArtifactClassification = {
  type: "image",
  icon: Image01Icon,
  label: "Image",
  openable: true,
  hasSourceToggle: false,
};

const VIDEO_CLASSIFICATION: ArtifactClassification = {
  type: "video",
  icon: Video01Icon,
  label: "Video",
  openable: true,
  hasSourceToggle: false,
};

// The media URL answers with an error status, so the element's own onError
// fires and the component swaps in its failure message. The path match ignores
// the `_retry` query that Try again adds, so a retry fails the same way.
function fails(artifact: ArtifactRef, status: number) {
  return http.get(
    `*${artifact.sourceUrl}`,
    () => new HttpResponse(null, { status }),
  );
}

const meta: Meta<typeof ArtifactContent> = {
  title: "Copilot/ArtifactContent errors",
  component: ArtifactContent,
  parameters: {
    layout: "padded",
    docs: {
      description: {
        component:
          "Image and video artifacts whose media request fails: the preview shows a failure message and a Try again button.",
      },
    },
  },
  decorators: [
    (Story) => (
      <div className="flex h-[500px] w-[600px] max-w-full flex-col overflow-hidden border border-border-base">
        <Story />
      </div>
    ),
  ],
};

export default meta;
type Story = StoryObj<typeof meta>;

export const ImageFailed: Story = {
  name: "Image failed to load",
  args: {
    artifact: IMAGE_ARTIFACT,
    isSourceView: false,
    classification: IMAGE_CLASSIFICATION,
  },
  parameters: { msw: { handlers: [fails(IMAGE_ARTIFACT, 404)] } },
};

export const VideoFailed: Story = {
  name: "Video failed to load",
  args: {
    artifact: VIDEO_ARTIFACT,
    isSourceView: false,
    classification: VIDEO_CLASSIFICATION,
  },
  parameters: { msw: { handlers: [fails(VIDEO_ARTIFACT, 500)] } },
};
