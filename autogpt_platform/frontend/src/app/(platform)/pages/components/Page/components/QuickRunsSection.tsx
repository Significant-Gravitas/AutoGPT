import { usePostV2ExecuteAPreset } from "@/app/api/__generated__/endpoints/presets/presets";
import type { PageQuickRuns } from "@/app/api/__generated__/models/pageQuickRuns";
import { okData } from "@/app/api/helpers";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { trackPageRunStarted } from "@/services/pages/pages-analytics";
import { PlayIcon } from "@hugeicons/core-free-icons";
import { useState } from "react";
import type { PageLive } from "../Page";

interface Props {
  section: PageQuickRuns;
  live?: PageLive;
}

export function QuickRunsSection({ section, live }: Props) {
  const [pending, setPending] = useState<string | null>(null);
  const { toast } = useToast();
  const execute = usePostV2ExecuteAPreset();

  if (!section.templates.length) {
    return (
      <Text variant="body" tone="secondary">
        No saved templates for this workflow yet.
      </Text>
    );
  }

  async function run(templateID: string) {
    if (!live) return;
    setPending(templateID);
    try {
      const started = okData(
        await execute.mutateAsync({ presetId: templateID, data: {} }),
      );
      if (!started) throw new Error("The run did not start");
      trackPageRunStarted("quick_run");
      live.onRunStarted(section.library_agent_id, started.id);
    } catch {
      toast({ title: "Couldn't start the run", variant: "destructive" });
    } finally {
      setPending(null);
    }
  }

  return (
    <div className="flex flex-wrap gap-2">
      {section.templates.map((template) => (
        <Button
          key={template.id}
          size="small"
          variant="outline"
          disabled={!live || pending !== null}
          loading={pending === template.id}
          leftIcon={<Icon icon={PlayIcon} size={14} />}
          onClick={() => run(template.id)}
        >
          {template.name}
        </Button>
      ))}
    </div>
  );
}
