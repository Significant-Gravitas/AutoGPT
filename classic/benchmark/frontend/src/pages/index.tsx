import { useEffect, useState } from "react";
import Head from "next/head";
import tw from "tailwind-styled-components";

import Graph from "../components/index/Graph";
import TaskInfo from "../components/index/TaskInfo";
import { TaskData } from "../lib/types";

const Home = () => {
  const [data, setData] = useState(null);
  const [selectedTask, setSelectedTask] = useState<TaskData | null>(null);
  const [isTaskInfoExpanded, setIsTaskInfoExpanded] = useState(false);

  useEffect(() => {
    // Load the JSON data from the public folder
    fetch("/graph.json")
      .then((response) => response.json())
      .then((data) => {
        setData(data);
      })
      .catch((error) => {
        console.error("Error fetching the graph data:", error);
      });
  }, []);

  return (
    <>
      <Head>
        <title>agbenchmark</title>
        <meta
          name="description"
          content="The best way to evaluate your agents"
        />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      <main className="flex h-screen flex-col items-center justify-center">
        {data && (
          <Panels>
            <Graph
              graphData={data}
              setSelectedTask={setSelectedTask}
              setIsTaskInfoExpanded={setIsTaskInfoExpanded}
            />
            <TaskInfo
              selectedTask={selectedTask}
              isTaskInfoExpanded={isTaskInfoExpanded}
              setIsTaskInfoExpanded={setIsTaskInfoExpanded}
              setSelectedTask={setSelectedTask}
            />
          </Panels>
        )}
      </main>
    </>
  );
};

export default Home;

const Panels = tw.div`
  flex
  h-full
  w-full
`;
