import React, { createContext, useContext, useEffect } from 'react';
import logPageViewAction from './actions';

const PageViewContext = createContext<null>(null);

export const PageViewProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {

  useEffect(() => {
    const logPageView = async () => {
      const pageViewData = { page: window.location.pathname, data: {} };
      await logPageViewAction(pageViewData.page, pageViewData.data);
    };

    logPageView().catch(console.error);

  }, []);

  return (
    <PageViewContext.Provider value={null}>
      {children}
    </PageViewContext.Provider>
  );
};

export const usePageViews = () => useContext(PageViewContext);