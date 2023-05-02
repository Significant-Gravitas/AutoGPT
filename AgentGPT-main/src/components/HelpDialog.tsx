import React from "react";
import { useTranslation } from "next-i18next";
import { FaDiscord, FaGithub, FaTwitter } from "react-icons/fa";
import Dialog from "./Dialog";

export default function HelpDialog({
  show,
  close,
}: {
  show: boolean;
  close: () => void;
}) {
  const [ t ] = useTranslation();
  return (
    <Dialog header={`${t('WELCOME_TO_CHATGPT')} 🤖`} isShown={show} close={close}>
      <div className="text-md relative flex-auto p-2 leading-relaxed">
        <p>
          <strong>AgentGPT</strong> {t('allows you to configure and deploy Autonomous AI agents. Name your custom AI and have it embark on any goal imaginable. It will attempt to reach the goal by thinking of tasks to do, executing them, and learning from the results 🚀')}
        </p>
        <div>
          <br />
          {t('This platform is currently in beta, we are currently working on:')}
          <ul className="ml-5 list-inside list-disc">
            <li>{`${t('LONG_TERM_MEMORY')} 🧠`}</li>
            <li>{`${t('WEB_BROWSING')} 🌐`}</li>
            <li>{`${t('INTERACTION_WITH_WEBSITES_AND_PEOPLE')} 👨‍👩‍👦`}</li>
          </ul>
          <br />
          <p className="mt-2">{t('Follow the journey below:')}</p>
        </div>
        <div className="mt-4 flex w-full items-center justify-center gap-5">
          <div
            className="cursor-pointer rounded-full bg-black/30 p-3 hover:bg-black/70"
            onClick={() =>
              window.open("https://discord.gg/jdSBAnmdnY", "_blank")
            }
          >
            <FaDiscord size={30} />
          </div>
          <div
            className="cursor-pointer rounded-full bg-black/30 p-3 hover:bg-black/70"
            onClick={() =>
              window.open(
                "https://twitter.com/asimdotshrestha/status/1644883727707959296",
                "_blank"
              )
            }
          >
            <FaTwitter size={30} />
          </div>
          <div
            className="cursor-pointer rounded-full bg-black/30 p-3 hover:bg-black/70"
            onClick={() =>
              window.open("https://github.com/reworkd/AgentGPT", "_blank")
            }
          >
            <FaGithub size={30} />
          </div>
        </div>
      </div>
    </Dialog>
  );
}
