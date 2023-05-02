import React, { useEffect, useRef } from "react";
import { useTranslation } from "next-i18next";
import { type NextPage, type GetStaticProps } from "next";
import Badge from "../components/Badge";
import DefaultLayout from "../layout/default";
import ChatWindow from "../components/ChatWindow";
import Drawer from "../components/Drawer";
import Input from "../components/Input";
import Button from "../components/Button";
import { FaRobot, FaStar } from "react-icons/fa";
import PopIn from "../components/motions/popin";
import { VscLoading } from "react-icons/vsc";
import AutonomousAgent from "../components/AutonomousAgent";
import Expand from "../components/motions/expand";
import HelpDialog from "../components/HelpDialog";
import { SettingsDialog } from "../components/SettingsDialog";
import { TaskWindow } from "../components/TaskWindow";
import { useAuth } from "../hooks/useAuth";
import type { Message } from "../types/agentTypes";
import { useAgent } from "../hooks/useAgent";
import { isEmptyOrBlank } from "../utils/whitespace";
import { serverSideTranslations } from "next-i18next/serverSideTranslations";
import { useSettings } from "../hooks/useSettings";

const Home: NextPage = () => {
  const [t] = useTranslation();
  const { session, status } = useAuth();
  const [name, setName] = React.useState<string>("");
  const [goalInput, setGoalInput] = React.useState<string>("");
  const [agent, setAgent] = React.useState<AutonomousAgent | null>(null);
  const { settings, saveSettings } = useSettings();
  const [shouldAgentStop, setShouldAgentStop] = React.useState(false);
  const [messages, setMessages] = React.useState<Message[]>([]);
  const [showHelpDialog, setShowHelpDialog] = React.useState(false);
  const [showSettingsDialog, setShowSettingsDialog] = React.useState(false);
  const [hasSaved, setHasSaved] = React.useState(false);
  const agentUtils = useAgent();

  useEffect(() => {
    const key = "agentgpt-modal-opened-new";
    const savedModalData = localStorage.getItem(key);

    // Momentarily always run
    setTimeout(() => {
      if (savedModalData == null) {
        setShowHelpDialog(true);
      }
    }, 3000);

    localStorage.setItem(key, JSON.stringify(true));
  }, []);

  const nameInputRef = useRef<HTMLInputElement>(null);
  useEffect(() => {
    nameInputRef?.current?.focus();
  }, []);

  useEffect(() => {
    if (agent == null) {
      setShouldAgentStop(false);
    }
  }, [agent]);

  const handleAddMessage = (message: Message) => {
    setMessages((prev) => [...prev, message]);
  };

  const tasks = messages.filter((message) => message.type === "task");

  const disableDeployAgent =
    agent != null || isEmptyOrBlank(name) || isEmptyOrBlank(goalInput);

  const handleNewGoal = () => {
    const agent = new AutonomousAgent(
      name.trim(),
      goalInput.trim(),
      handleAddMessage,
      () => setAgent(null),
      settings,
      session ?? undefined
    );
    setAgent(agent);
    setHasSaved(false);
    setMessages([]);
    agent.run().then(console.log).catch(console.error);
  };

  const handleKeyPress = (
    e:
      | React.KeyboardEvent<HTMLInputElement>
      | React.KeyboardEvent<HTMLTextAreaElement>
  ) => {
    if (e.key === "Enter" && !disableDeployAgent) {
      if (!e.shiftKey) {
        // Only Enter is pressed, execute the function
        handleNewGoal();
      }
    }
  };

  const handleStopAgent = () => {
    setShouldAgentStop(true);
    agent?.stopAgent();
  };

  const proTitle = (
    <>
      AgentGPT<span className="ml-1 text-amber-500/90">Pro</span>
    </>
  );

  const shouldShowSave =
    status === "authenticated" &&
    !agent?.isRunning &&
    messages.length &&
    !hasSaved;

  return (
    <DefaultLayout>
      <HelpDialog
        show={showHelpDialog}
        close={() => setShowHelpDialog(false)}
      />
      <SettingsDialog
        customSettings={[settings, saveSettings]}
        show={showSettingsDialog}
        close={() => setShowSettingsDialog(false)}
      />
      <main className="flex min-h-screen flex-row">
        <Drawer
          showHelp={() => setShowHelpDialog(true)}
          showSettings={() => setShowSettingsDialog(true)}
        />
        <div
          id="content"
          className="z-10 flex min-h-screen w-full items-center justify-center p-2 px-2 sm:px-4 md:px-10"
        >
          <div
            id="layout"
            className="flex h-full w-full max-w-screen-lg flex-col items-center justify-between gap-3 py-5 md:justify-center"
          >
            <div
              id="title"
              className="relative flex flex-col items-center font-mono"
            >
              <div className="flex flex-row items-start shadow-2xl">
                <span className="text-4xl font-bold text-[#C0C0C0] xs:text-5xl sm:text-6xl">
                  Agent
                </span>
                <span className="text-4xl font-bold text-white xs:text-5xl sm:text-6xl">
                  GPT
                </span>
                <PopIn delay={0.5} className="sm:absolute sm:right-0 sm:top-2">
                  <Badge>Beta 🚀</Badge>
                </PopIn>
              </div>
              <div className="mt-1 text-center font-mono text-[0.7em] font-bold text-white">
                <p>
                  {t(
                    "Assemble, configure, and deploy autonomous AI Agents in your browser."
                  )}
                </p>
              </div>
            </div>

            <Expand className="flex w-full flex-row">
              <ChatWindow
                className="sm:mt-4"
                messages={messages}
                title={session?.user.subscriptionId ? proTitle : "AgentGPT"}
                showDonation={
                  status != "loading" && !session?.user.subscriptionId
                }
                onSave={
                  shouldShowSave
                    ? (format) => {
                        setHasSaved(true);
                        agentUtils.saveAgent({
                          goal: goalInput.trim(),
                          name: name.trim(),
                          tasks: messages,
                        });
                      }
                    : undefined
                }
                scrollToBottom
              />
              {tasks.length > 0 && <TaskWindow tasks={tasks} />}
            </Expand>

            <div className="flex w-full flex-col gap-2 sm:mt-4 md:mt-10">
              <Expand delay={1.2}>
                <Input
                  inputRef={nameInputRef}
                  left={
                    <>
                      <FaRobot />
                      <span className="ml-2">{t("AGENT_NAME")}</span>
                    </>
                  }
                  value={name}
                  disabled={agent != null}
                  onChange={(e) => setName(e.target.value)}
                  onKeyDown={(e) => handleKeyPress(e)}
                  placeholder="AgentGPT"
                  type="text"
                />
              </Expand>
              <Expand delay={1.3}>
                <Input
                  left={
                    <>
                      <FaStar />
                      <span className="ml-2">{t("AGENT_GOAL")}</span>
                    </>
                  }
                  disabled={agent != null}
                  value={goalInput}
                  onChange={(e) => setGoalInput(e.target.value)}
                  onKeyDown={(e) => handleKeyPress(e)}
                  placeholder={`${t("Make the world a better place.")}`}
                  type="textarea"
                />
              </Expand>
            </div>
            <Expand delay={1.4} className="flex gap-2">
              <Button
                disabled={disableDeployAgent}
                onClick={handleNewGoal}
                className="sm:mt-10"
              >
                {agent == null ? (
                  t("Deploy Agent")
                ) : (
                  <>
                    <VscLoading className="animate-spin" size={20} />
                    <span className="ml-2">{t("Running")}</span>
                  </>
                )}
              </Button>
              <Button
                disabled={agent == null}
                onClick={handleStopAgent}
                className="sm:mt-10"
                enabledClassName={"bg-red-600 hover:bg-red-400"}
              >
                {shouldAgentStop ? (
                  <>
                    <VscLoading className="animate-spin" size={20} />
                    <span className="ml-2">{t("Stopping")}</span>
                  </>
                ) : (
                  t("Stop Agent")
                )}
              </Button>
            </Expand>
          </div>
        </div>
      </main>
    </DefaultLayout>
  );
};

export default Home;

export const getStaticProps: GetStaticProps = async ({ locale = "en" }) => {
  const supportedLocales = [
    "en",
    "hu",
    "fr",
    "de",
    "it",
    "ja",
    "zh",
    "ko",
    "pl",
    "pt",
    "ro",
    "ru",
    "uk",
    "es",
    "nl",
    "sk",
    "hr",
  ];
  const chosenLocale = supportedLocales.includes(locale) ? locale : "en";

  return {
    props: {
      ...(await serverSideTranslations(chosenLocale, ["translation"])),
    },
  };
};
