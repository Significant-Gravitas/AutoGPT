import { type ReactNode } from "react";
import Head from "next/head";
import DottedGridBackground from "../components/DottedGridBackground";
import clsx from "clsx";
import { useTranslation } from 'react-i18next';

interface LayoutProps {
  children: ReactNode;
  className?: string;
  centered?: boolean;
}

const DefaultLayout = (props: LayoutProps) => {
  const [ t ] = useTranslation();
  const description =
    t('Assemble, configure, and deploy autonomous AI Agents in your browser.');
  return (
    <div
      className={clsx(
        "flex flex-col bg-gradient-to-b from-[#2B2B2B] to-[#1F1F1F]",
        props.centered && "items-center justify-center"
      )}
    >
      <Head>
        <title>AgentGPT</title>
        <meta name="description" content={description} />
        <meta name="twitter:site" content="@AgentGPT" />
        <meta name="twitter:card" content="summary_large_image" />
        <meta name="twitter:title" content="AgentGPT 🤖" />
        <meta name="twitter:description" content={description} />
        <meta
          name="twitter:image"
          content="https://agentgpt.reworkd.ai/banner.png"
        />
        <meta name="twitter:image:width" content="1280" />
        <meta name="twitter:image:height" content="640" />
        <meta
          property="og:title"
          content="AgentGPT: Autonomous AI in your browser 🤖"
        />
        <meta
          property="og:description"
          content="Assemble, configure, and deploy autonomous AI Agents in your browser."
        />
        <meta property="og:url" content="https://agentgpt.reworkd.ai/" />
        <meta
          property="og:image"
          content="https://agentgpt.reworkd.ai/banner.png"
        />
        <meta property="og:image:width" content="1280" />
        <meta property="og:image:height" content="640" />
        <meta property="og:type" content="website" />
        <meta
          name="google-site-verification"
          content="sG4QDkC8g2oxKSopgJdIe2hQ_SaJDaEaBjwCXZNkNWA"
        />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      <DottedGridBackground
        className={clsx("min-w-screen min-h-screen", props.className)}
      >
        {props.children}
      </DottedGridBackground>
    </div>
  );
};

export default DefaultLayout;
