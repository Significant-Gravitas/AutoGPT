import Link from "next/link";
import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { Text } from "@/components/atoms/Text/Text";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import { useExpertMap } from "../../copilot/useExpertMap";

export function MobileExperts() {
  const {
    activeExperts,
    isLoadingExperts,
    hasExpertsErrored,
    isExpertsEnabled,
  } = useExpertMap();
  if (isLoadingExperts) return <LoadingSpinner />;
  if (hasExpertsErrored)
    return (
      <Text variant="body" role="alert">
        {"We couldn't load your experts. Reload to try again."}
      </Text>
    );
  if (!isExpertsEnabled)
    return (
      <Text variant="body">
        {"Experts aren't available in this workspace yet."}
      </Text>
    );
  return (
    <section aria-label="Your experts" className="flex flex-col gap-4">
      {activeExperts.length === 0 ? (
        <Text variant="body" tone="secondary">
          Hire an expert to start a conversation.
        </Text>
      ) : (
        <ul className="divide-y divide-zinc-100 overflow-hidden rounded-2xl border border-zinc-200 bg-white">
          {activeExperts.map((expert) => (
            <li key={expert.id}>
              <Link
                href={`/home?expertId=${encodeURIComponent(expert.id)}`}
                aria-label={`Chat with ${expert.name}`}
                className="flex min-h-20 items-center gap-3 px-4 py-3 hover:bg-zinc-50 focus-visible:bg-zinc-50"
              >
                <ExpertAvatar
                  name={expert.name}
                  avatarUrl={expert.avatarUrl}
                  size={48}
                />
                <div className="min-w-0">
                  <Text variant="body-medium" unmask={false}>
                    {expert.name}
                  </Text>
                  <Text variant="small" tone="secondary" unmask={false}>
                    {expert.jobTitle || expert.role}
                  </Text>
                </div>
              </Link>
            </li>
          ))}
        </ul>
      )}
      <Link
        href="/marketplace"
        className="flex min-h-11 items-center justify-center rounded-full border border-zinc-200 bg-white px-5 text-sm font-medium"
      >
        Find an expert
      </Link>
    </section>
  );
}
