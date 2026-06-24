import { TooltipProvider } from "@/components/atoms/Tooltip/BaseTooltip";
import { BackendAPIProvider } from "@/lib/autogpt-server-api/context";
import OnboardingProvider from "@/providers/onboarding/onboarding-provider";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, RenderOptions } from "@testing-library/react";
import { NuqsTestingAdapter } from "nuqs/adapters/testing";
import { ReactElement, ReactNode } from "react";

function createTestQueryClient() {
  return new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
      },
    },
  });
}

function TestProviders({ children }: { children: ReactNode }) {
  const queryClient = createTestQueryClient();
  return (
    <QueryClientProvider client={queryClient}>
      <NuqsTestingAdapter>
        <BackendAPIProvider>
          <OnboardingProvider>
            <TooltipProvider>{children}</TooltipProvider>
          </OnboardingProvider>
        </BackendAPIProvider>
      </NuqsTestingAdapter>
    </QueryClientProvider>
  );
}

function customRender(
  ui: ReactElement,
  options?: Omit<RenderOptions, "wrapper">,
) {
  return render(ui, { wrapper: TestProviders, ...options });
}

// MorphingTextAnimation renders one span per character and uses a non-breaking
// space for spaces, so normalize all whitespace before asserting on the text.
function normalizeWhitespace(container: HTMLElement): string {
  return (container.textContent ?? "").replace(/\s+/g, " ").trim();
}

export * from "@testing-library/react";
export { customRender as render, normalizeWhitespace };
