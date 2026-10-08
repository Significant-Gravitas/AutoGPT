import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { Text } from "@/components/atoms/Text/Text";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { NeedsYou } from "../../home/components/NeedsYou/NeedsYou";
import { useHomePage } from "../../home/useHomePage";

export function MobileAttention() {
  const { dashboard, isLoading, isError, refetch } = useHomePage({
    enabled: true,
  });
  if (isLoading) return <LoadingSpinner />;
  if (isError || !dashboard)
    return (
      <ErrorCard
        context="requests that need your attention"
        onRetry={() => refetch()}
      />
    );
  return (
    <section
      aria-label="Requests that need your attention"
      className="flex flex-col gap-4"
    >
      <Text variant="body" tone="secondary">
        Answer questions, review approvals, and finish setup here. Open a
        question to respond in its conversation.
      </Text>
      {dashboard.attention.length ? (
        <NeedsYou dashboard={dashboard} />
      ) : (
        <Text variant="body">{"You're all caught up."}</Text>
      )}
    </section>
  );
}
