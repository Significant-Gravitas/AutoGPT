import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";

export function EmptyTeamState() {
  return (
    <div className="flex flex-col items-center gap-3 rounded-xl border border-dashed border-zinc-200 bg-white px-6 py-10 text-center">
      <Text variant="body-medium">No hired experts yet</Text>
      <Text variant="small" className="max-w-prose text-zinc-500">
        Hire an expert from the marketplace and they will show up here, ready to
        work alongside Autopilot.
      </Text>
      <div className="mt-1 flex flex-wrap items-center justify-center gap-2">
        <Button
          as="NextLink"
          href="/marketplace"
          variant="primary"
          size="small"
        >
          Browse the marketplace
        </Button>
        <Button as="NextLink" href="/raise" variant="secondary" size="small">
          Raise your own
        </Button>
      </div>
    </div>
  );
}
