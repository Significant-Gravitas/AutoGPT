import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { Badge } from "@/components/atoms/Badge/Badge";
import { Link } from "@/components/atoms/Link/Link";
import { Text } from "@/components/atoms/Text/Text";
import { ShowMoreText } from "@/components/molecules/ShowMoreText/ShowMoreText";
import { humanizeCronExpression } from "@/lib/cron-expression-utils";

interface ModalHeaderProps {
  agent: LibraryAgent;
}

export function ModalHeader({ agent }: ModalHeaderProps) {
  const creator = agent.marketplace_listing?.creator;

  return (
    <div className="flex flex-col gap-4">
      <Badge variant="info" className="w-fit">
        New Task
      </Badge>
      <div>
        <Text variant="h2">{agent.name}</Text>
        {creator ? (
          <Link href={`/marketplace/creator/${creator.slug}`} isExternal>
            by {creator.name}
          </Link>
        ) : null}

        {agent.description ? (
          <ShowMoreText
            previewLimit={400}
            variant="small"
            className="mb-2 mt-4 !text-foreground-body"
          >
            {agent.description}
          </ShowMoreText>
        ) : null}

        {agent.recommended_schedule_cron && !agent.has_external_trigger ? (
          <div
            // TODO(tokens): no exact role for border-blue-100 (#DBEAFE)
            className="flex flex-col gap-4 rounded-medium border border-blue-100 bg-info-surface p-4"
          >
            <Text variant="lead-semibold" className="text-info-foreground">
              Tip
            </Text>
            <div
              // TODO(tokens): no exact role for bg-blue-100 (#DBEAFE)
              className="h-px w-full bg-blue-100"
            />

            <Text variant="body">
              For best results, run this workflow{" "}
              {humanizeCronExpression(
                agent.recommended_schedule_cron,
              ).toLowerCase()}
            </Text>
          </div>
        ) : null}

        {agent.instructions ? (
          <div
            // TODO(tokens): no exact role for border-purple-100 (#EFE8FE); no exact role for bg-[#f1ebfe80]
            className="mt-4 flex flex-col gap-4 rounded-medium border border-purple-100 bg-[#f1ebfe80] p-4"
          >
            <Text
              // TODO(tokens): no fitting role for text-purple-600 #6C2EDF; #6C2EDF is brand-hover, a hover fill
              variant="lead-semibold"
              className="text-purple-600"
            >
              Instructions
            </Text>

            <div
              // TODO(tokens): no exact role for bg-purple-100 (#EFE8FE)
              className="h-px w-full bg-purple-100"
            />

            <Text variant="body">{agent.instructions}</Text>
          </div>
        ) : null}
      </div>
    </div>
  );
}
