import React, { useState } from "react";

import tw from "tailwind-styled-components";

import { TaskData } from "../../lib/types";
import RunData from "./RunData";
import SelectedTask from "./SelectedTask";
import MockCheckbox from "./MockCheckbox";
import RunButton from "./RunButton";

interface TaskInfoProps {
  selectedTask: TaskData | null;
  isTaskInfoExpanded: boolean;
  setIsTaskInfoExpanded: React.Dispatch<React.SetStateAction<boolean>>;
  setSelectedTask: React.Dispatch<React.SetStateAction<TaskData | null>>;
}

const TaskInfo: React.FC<TaskInfoProps> = ({
  selectedTask,
  isTaskInfoExpanded,
  setIsTaskInfoExpanded,
  setSelectedTask,
}) => {
  const [isMock, setIsMock] = useState<boolean>(false);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [allResponseData, setAllResponseData] = useState<any[]>([]);
  const [responseData, setResponseData] = useState<any>();
  const [cutoff, setCutoff] = useState<number | null>(null);

  const runBenchmark = async () => {
    setIsLoading(true);
    try {
      let url = `http://localhost:8000/run?mock=${isMock}`;
      cutoff && !isMock && (url += `&cutoff=${cutoff}`);
      const response = await fetch(url);
      const data = await response.json();

      if (data["returncode"] > 0) {
        throw new Error(data["stderr"]);
      } else {
        const jsonObject = JSON.parse(data["stdout"]);
        setAllResponseData([...allResponseData, jsonObject]);
        setResponseData(jsonObject);
      }
    } catch (error) {
      console.error("There was an error fetching the data", error);
    }
    setIsLoading(false);
  };

  return (
    <TaskDetails isExpanded={isTaskInfoExpanded}>
      {isTaskInfoExpanded ? (
        <ToggleButton
          onClick={() => {
            setIsTaskInfoExpanded(!isTaskInfoExpanded);
            setSelectedTask(null);
          }}
        >
          →
        </ToggleButton>
      ) : (
        <BenchmarkWrapper>
          <RunButton
            cutoff={selectedTask?.cutoff}
            isLoading={isLoading}
            testRun={runBenchmark}
            isMock={isMock}
          />
          <MockCheckbox isMock={isMock} setIsMock={setIsMock} />
          <Detail>
            <b>or click a node on the left</b>
          </Detail>
        </BenchmarkWrapper>
      )}

      {selectedTask && (
        <SelectedTask
          selectedTask={selectedTask}
          isMock={isMock}
          setIsMock={setIsMock}
          cutoff={cutoff}
          setResponseData={setResponseData}
          allResponseData={allResponseData}
          setAllResponseData={setAllResponseData}
        />
      )}
      {!isMock && (
        <CheckboxWrapper>
          <p>Custom cutoff</p>
          <CutoffInput
            type="number"
            placeholder="Leave blank for default"
            value={cutoff ?? ""}
            onChange={(e) =>
              setCutoff(e.target.value ? parseInt(e.target.value) : null)
            }
          />
        </CheckboxWrapper>
      )}
      <Header>Previous Run</Header>
      {!responseData && <p>No runs yet</p>}
      {responseData && <RunData latestRun={responseData} />}
      <Header>All Runs</Header>
      {allResponseData.length === 0 && <p>No runs yet</p>}
      {allResponseData.length > 1 &&
        allResponseData
          .slice(0, -1)
          .map((responseData, index) => (
            <RunData key={index} latestRun={responseData} />
          ))}
    </TaskDetails>
  );
};

export default TaskInfo;

const TaskDetails = tw.div<{ isExpanded: boolean }>`
  ${(p) => (p.isExpanded ? "w-1/2" : "w-1/4")}
  ml-5
  transition-all
  duration-500
  ease-in-out
  p-4
  border
  border-gray-400
  h-full
  overflow-x-hidden
`;

const Header = tw.h5`
  text-xl
  font-semibold
  mt-4
`;

const ToggleButton = tw.button`
    font-bold
    text-2xl
`;

const BenchmarkWrapper = tw.div`
    flex
    flex-col
    items-center
    justify-center
`;

const CutoffInput = tw.input`
  border rounded w-1/2 h-8 text-sm
  focus:outline-none focus:border-blue-400
  pl-2
`;

const Detail = tw.p`
    mt-2
`;

const CheckboxWrapper = tw.label`
    flex 
    items-center 
    space-x-2 
    mt-2
`;
