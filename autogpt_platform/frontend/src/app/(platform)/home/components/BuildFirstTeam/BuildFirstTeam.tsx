import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";

const FUNCTIONS = ["Product", "Engineering", "Marketing", "Sales"];

export function BuildFirstTeam() {
  return (
    <section className="mx-auto mt-8 flex w-full max-w-3xl flex-col items-center rounded-[30px] bg-white px-6 py-12 text-center shadow-zinc-950 smooth-shadow-ring-sm sm:px-10">
      <Text variant="h3" className="text-zinc-950">
        Build your first AI team
      </Text>
      <Text variant="large" className="mt-3 max-w-xl text-pretty text-zinc-600">
        Tell AutoPilot what you are building. It will propose the smallest team,
        ask for one approval, and put each expert to work.
      </Text>
      <div className="mt-6 flex flex-wrap justify-center gap-2">
        {FUNCTIONS.map((name) => (
          <span
            key={name}
            className="rounded-full bg-zinc-100 px-3 py-1.5 text-sm font-medium text-zinc-700"
          >
            {name}
          </span>
        ))}
      </div>
      <Button
        as="NextLink"
        href="/copilot"
        variant="primary"
        size="large"
        className="mt-7"
      >
        Start with AutoPilot
      </Button>
    </section>
  );
}
