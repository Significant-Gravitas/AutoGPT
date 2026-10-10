import { ToolChain } from "../../../components/ToolChain/ToolChain";
import type { RenderUIMessagePart } from "../isRenderUIPart";

interface Props {
  part: RenderUIMessagePart;
  message: string;
  readOnly: boolean;
}

export function RejectedUI({ part, message, readOnly }: Props) {
  return (
    <ToolChain
      parts={[
        {
          type: "tool-render_ui",
          toolCallId: part.toolCallId,
          state: "output-available",
          input: part.input,
          output: { type: "error", message },
        },
      ]}
      isStreaming={false}
      readOnly={readOnly}
    />
  );
}
