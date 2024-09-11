import React, { createContext, useContext, useEffect } from "react";
import { usePathname, useSearchParams } from "next/navigation";
import logPageViewAction from "./actions";

const EXCLUDED_PATHS = ["/login"];

const PageViewContext = createContext<null>(null);

export const PageViewProvider: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => {
  const pathname = usePathname();
  const searchParams = useSearchParams();

  useEffect(() => {
    if (EXCLUDED_PATHS.includes(pathname)) {
      return;
    }

    const logPageView = async () => {
      const pageViewData = {
        page: pathname,
        data: Object.fromEntries(searchParams.entries()),
      };
      await logPageViewAction(pageViewData.page, pageViewData.data);
    };

    logPageView().catch(console.error);
  }, [pathname, searchParams]);

  return (
    <PageViewContext.Provider value={null}>{children}</PageViewContext.Provider>
  );
};

export const usePageViews = () => useContext(PageViewContext);
