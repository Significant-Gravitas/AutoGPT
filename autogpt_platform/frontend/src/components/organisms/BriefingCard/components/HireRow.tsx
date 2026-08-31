import { useQueryClient } from "@tanstack/react-query";
import { useHireExpert } from "@/app/api/__generated__/endpoints/experts/experts";
import type { BriefingHireItem } from "@/app/api/__generated__/models/briefingHireItem";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { toast } from "@/components/molecules/Toast/use-toast";
import { invalidateExpertRosterQueries } from "@/services/experts/invalidate-experts";

interface Props {
  item: BriefingHireItem;
}

// A recommendation to hire a template Autopilot has been covering: one-tap
// hire, with the roster refreshed so the new teammate shows up right away.
export function HireRow({ item }: Props) {
  const queryClient = useQueryClient();
  const { mutate: hire, isPending } = useHireExpert({
    mutation: {
      onSuccess: async () => {
        toast({ title: `${item.name} joined your team` });
        await invalidateExpertRosterQueries(queryClient);
      },
      onError: () =>
        toast({
          title: `Couldn't hire ${item.name}`,
          description: "Please try again.",
          variant: "destructive",
        }),
    },
  });

  return (
    <li className="flex items-center gap-3 px-5 py-4">
      <div className="min-w-0 flex-1">
        <Text variant="body-medium" className="truncate text-zinc-900">
          {item.name} — {item.role}
        </Text>
        <Text variant="body" className="text-zinc-500">
          Autopilot handled {item.task_count} similar{" "}
          {item.task_count === 1 ? "task" : "tasks"}
        </Text>
      </div>
      <Button
        variant="primary"
        size="small"
        loading={isPending}
        onClick={() => hire({ data: { template_id: item.template_id } })}
      >
        Hire
      </Button>
    </li>
  );
}
