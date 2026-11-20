import { Image01Icon, Video01Icon } from "@hugeicons/core-free-icons";
import type { Meta, StoryObj } from "@storybook/nextjs";
import { delay, http, HttpResponse } from "msw";
import type { ArtifactRef } from "../../../store";
import type { ArtifactClassification } from "../helpers";
import { ArtifactContent } from "./ArtifactContent";

const PROXY_BASE = "/api/proxy/api/workspace/files";

const IMAGE_ARTIFACT: ArtifactRef = {
  id: "loading-image",
  title: "chart.png",
  mimeType: "image/png",
  sourceUrl: `${PROXY_BASE}/loading-image/download`,
  origin: "agent",
};

const VIDEO_ARTIFACT: ArtifactRef = {
  id: "loading-video",
  title: "walkthrough.mp4",
  mimeType: "video/mp4",
  sourceUrl: `${PROXY_BASE}/loading-video/download`,
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

// The media request never answers, so the element stays in its loading state
// and the skeleton over it never goes away.
function neverLoads(artifact: ArtifactRef) {
  return http.get(`*${artifact.sourceUrl}`, async () => {
    await delay("infinite");
    return new HttpResponse(null, { status: 204 });
  });
}

// The first bytes a slow download of an 800x500 PNG delivers: the signature,
// the IHDR chunk and the start of the first IDAT chunk. That is enough for the
// browser to learn the image's size and lay the <img> out at it, but the image
// never finishes loading.
const PNG_HEAD_800X500 = new Uint8Array([
  // Signature
  137, 80, 78, 71, 13, 10, 26, 10,
  // IHDR: 800x500, 8-bit RGB
  0, 0, 0, 13, 73, 72, 68, 82, 0, 0, 3, 32, 0, 0, 1, 244, 8, 2, 0, 0, 0, 74, 34,
  222, 47,
  // IDAT length and type, its data still on the way
  0, 0, 16, 0, 73, 68, 65, 84,
]);

function stallsAfterHeader(artifact: ArtifactRef) {
  return http.get(`*${artifact.sourceUrl}`, () => {
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(PNG_HEAD_800X500);
      },
    });
    return new HttpResponse(body, { headers: { "Content-Type": "image/png" } });
  });
}

const meta: Meta<typeof ArtifactContent> = {
  title: "Copilot/ArtifactContent loading",
  component: ArtifactContent,
  parameters: {
    layout: "padded",
    docs: {
      description: {
        component:
          "Image and video artifacts while the media is still loading: a skeleton covers the element until it reports it has loaded.",
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

export const ImageLoading: Story = {
  name: "Image loading (skeleton)",
  args: {
    artifact: IMAGE_ARTIFACT,
    isSourceView: false,
    classification: IMAGE_CLASSIFICATION,
  },
  parameters: { msw: { handlers: [stallsAfterHeader(IMAGE_ARTIFACT)] } },
};

export const VideoLoading: Story = {
  name: "Video loading (skeleton)",
  args: {
    artifact: VIDEO_ARTIFACT,
    isSourceView: false,
    classification: VIDEO_CLASSIFICATION,
  },
  parameters: { msw: { handlers: [neverLoads(VIDEO_ARTIFACT)] } },
};
