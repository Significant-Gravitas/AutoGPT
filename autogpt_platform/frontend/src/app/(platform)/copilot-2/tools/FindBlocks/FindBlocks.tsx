import { UIMessage, UIDataTypes, UITools, UIMessagePart } from "ai";

export const FindBlocksTool = ({
  message,
  i,
  part,
}: {
  message: UIMessage<unknown, UIDataTypes, UITools>;
  i: number;
  part: UIMessagePart<any, any>;
}) => {
  return (
    <div>
      <h1>Find Blocks</h1>
    </div>
  );
};
