import { type AppType } from "next/app";
import { type Session } from "next-auth";
import { SessionProvider } from "next-auth/react";
import { api } from "../utils/api";
import "../styles/globals.css";
import { Analytics } from "@vercel/analytics/react";
import { appWithTranslation, useTranslation } from "next-i18next";
import { useEffect } from "react";
import nextI18NextConfig from "../../next-i18next.config.js";

const MyApp: AppType<{ session: Session | null }> = ({
  Component,
  pageProps: { session, ...pageProps },
}) => {
  const { i18n } = useTranslation();

  useEffect(() => {
    i18n.on("languageChanged", () => {
      document.documentElement.lang = i18n.language;
    });
    document.documentElement.lang = i18n.language;
  }, [i18n]);

  return (
    <SessionProvider session={session}>
      <Component {...pageProps} />
      <Analytics />
    </SessionProvider>
  );
};

export default api.withTRPC(appWithTranslation(MyApp, nextI18NextConfig));
