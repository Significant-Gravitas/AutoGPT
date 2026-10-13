import { workspaceFileRenderer } from "@/components/contextual/OutputRenderers/renderers/WorkspaceFileRenderer";
import { imageRenderer } from "@/components/contextual/OutputRenderers/renderers/ImageRenderer";
import { videoRenderer } from "@/components/contextual/OutputRenderers/renderers/VideoRenderer";
import { audioRenderer } from "@/components/contextual/OutputRenderers/renderers/AudioRenderer";
import { csvRenderer } from "@/components/contextual/OutputRenderers/renderers/CSVRenderer";
import { jsonRenderer } from "@/components/contextual/OutputRenderers/renderers/JSONRenderer";
import { textRenderer } from "@/components/contextual/OutputRenderers/renderers/TextRenderer";

interface Props {
  outputs: Record<string, unknown[]>;
}
const renderers = [
  workspaceFileRenderer,
  videoRenderer,
  audioRenderer,
  imageRenderer,
  csvRenderer,
  jsonRenderer,
  textRenderer,
];
export function ExecutionOutputs({ outputs }: Props) {
  return (
    <div className="min-w-0 space-y-4">
      {Object.entries(outputs).map(([name, values]) => (
        <details key={name} open>
          <summary className="min-h-11 py-3 text-sm font-medium">
            {name} · {values.length} output{values.length === 1 ? "" : "s"}
          </summary>
          <div className="max-h-96 space-y-4 overflow-auto">
            {values.map((value, index) => (
              <div key={index}>
                {renderers
                  .find((renderer) => renderer.canRender(value))
                  ?.render(value)}
              </div>
            ))}
          </div>
        </details>
      ))}
    </div>
  );
}
