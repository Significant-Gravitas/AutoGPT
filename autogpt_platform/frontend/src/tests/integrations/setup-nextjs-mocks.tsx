import { createElement } from "react";
import { vi } from "vitest";
import type { ComponentPropsWithoutRef } from "react";

export const mockNextjsModules = () => {
  vi.mock("next/image", () => ({
    __esModule: true,
    default: ({
      fill: _fill,
      priority: _priority,
      quality: _quality,
      placeholder: _placeholder,
      blurDataURL: _blurDataURL,
      loader: _loader,
      ...props
    }: any) => {
      return createElement("img", props);
    },
  }));

  vi.mock("next/headers", () => ({
    cookies: vi.fn(() => ({
      get: vi.fn(() => undefined),
      getAll: vi.fn(() => []),
      set: vi.fn(),
      delete: vi.fn(),
      has: vi.fn(() => false),
    })),
    headers: vi.fn(() => new Headers()),
  }));

  vi.mock("next/dist/server/request/cookies", () => ({
    cookies: vi.fn(() => ({
      get: vi.fn(() => undefined),
      getAll: vi.fn(() => []),
      set: vi.fn(),
      delete: vi.fn(),
      has: vi.fn(() => false),
    })),
  }));

  vi.mock("next/navigation", () => ({
    useRouter: () => ({
      push: vi.fn(),
      replace: vi.fn(),
      prefetch: vi.fn(),
      back: vi.fn(),
      forward: vi.fn(),
      refresh: vi.fn(),
    }),
    usePathname: () => "/marketplace",
    useSearchParams: () => new URLSearchParams(),
    useParams: () => ({}),
    notFound: () => {
      throw new Error("NEXT_HTTP_ERROR_FALLBACK;404");
    },
  }));

  vi.mock("next/link", async () => {
    const { forwardRef } = await import("react");
    return {
      __esModule: true,
      default: forwardRef<HTMLAnchorElement, ComponentPropsWithoutRef<"a">>(
        function MockLink({ children, href, ...props }, ref) {
          return (
            <a ref={ref} href={href} {...props}>
              {children}
            </a>
          );
        },
      ),
      useLinkStatus: () => ({ pending: false }),
    };
  });
};
