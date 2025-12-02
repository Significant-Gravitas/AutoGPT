import { Button } from "@/components/atoms/Button/Button";

interface Props {
  flowId: string;
}

export function AgentCostSection({ flowId }: Props) {
  return (
    <div className="mt-6 flex items-center justify-between">
      {/* TODO: enable once we have an API to show estimated cost for an agent run */}
      {/* <div className="flex items-center gap-2">
        <Text variant="body-medium">Cost</Text>
        <Text variant="body">{cost}</Text>
      </div> */}
      <div className="flex items-center gap-2">
        <Button
          variant="outline"
          size="small"
          as="NextLink"
          href={`/build?flowID=${flowId}`}
        >
          Open in builder
        </Button>
        {/* TODO: enable once we can easily link to the agent listing page from the library agent response */}
        {/* <Button variant="outline" size="small">
          View listing <ArrowSquareOutIcon size={16} />
        </Button> */}
      </div>
    </div>
  );
}
