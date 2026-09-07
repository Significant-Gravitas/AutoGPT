import {
  configForName,
  type AvatarConfig,
  type AvatarStatus,
} from "@/components/molecules/BotAvatar/helpers";

export const QUERY_KEY = "a";

export interface Turn {
  yaw: number;
  pitch: number;
}

export const FRONT_TURN: Turn = { yaw: 0, pitch: 0 };

export function turnToPose(turn: Turn) {
  return {
    yaw: (turn.yaw * Math.PI) / 180,
    pitch: (turn.pitch * Math.PI) / 180,
  };
}
export const STAGE_SIZE = 220;
export const EXPORT_SIZE = 512;
export const SIZE_LADDER = [64, 40, 24] as const;

export const ROSTER_SAMPLE: {
  name: string;
  role: string;
  status: AvatarStatus;
}[] = [
  { name: "Otto", role: "AutoPilot", status: "idle" },
  { name: "Maya", role: "Marketing", status: "working" },
  { name: "Finn", role: "Finance", status: "waiting" },
  { name: "Sol", role: "Sales", status: "done" },
  { name: "Remy", role: "Research", status: "idle" },
  { name: "Cleo", role: "Content", status: "working" },
];

export function rosterConfigs() {
  return ROSTER_SAMPLE.map((member) => ({
    ...member,
    config: configForName(member.name),
  }));
}

export function serializeSvg(svg: SVGSVGElement) {
  const clone = svg.cloneNode(true) as SVGSVGElement;
  clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
  clone.setAttribute("width", String(EXPORT_SIZE));
  clone.setAttribute("height", String(EXPORT_SIZE));
  clone.removeAttribute("class");
  return new XMLSerializer().serializeToString(clone);
}

export function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

export function svgToPngBlob(markup: string) {
  return new Promise<Blob>((resolve, reject) => {
    const image = new Image();
    const source = URL.createObjectURL(
      new Blob([markup], { type: "image/svg+xml" }),
    );
    image.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = EXPORT_SIZE;
      canvas.height = EXPORT_SIZE;
      const context = canvas.getContext("2d");
      if (!context) {
        URL.revokeObjectURL(source);
        reject(new Error("Canvas unavailable"));
        return;
      }
      context.drawImage(image, 0, 0, EXPORT_SIZE, EXPORT_SIZE);
      URL.revokeObjectURL(source);
      canvas.toBlob(
        (blob) =>
          blob ? resolve(blob) : reject(new Error("PNG encode failed")),
        "image/png",
      );
    };
    image.onerror = () => {
      URL.revokeObjectURL(source);
      reject(new Error("SVG did not load"));
    };
    image.src = source;
  });
}

export function exportFilename(config: AvatarConfig, extension: "svg" | "png") {
  return `expert-avatar-${config.shape}-${config.color}-${config.accessory}.${extension}`;
}
