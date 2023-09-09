import React, { useState } from "react";
import tw from "tailwind-styled-components";

interface CurrentEnvProps {
  data: any;
}

const CurrentEnv: React.FC<CurrentEnvProps> = ({ data }) => {
  const [agentName, setAgentName] = useState<string>("mini-agi");
  const [reportLocation, setReportLocation] = useState<string>(
    "../reports/mini-agi"
  );
  const [openAiKey, setOpenAiKey] = useState<string>();

  return (
    <CurrentEnvContainer>
      <Title>Env Variables</Title>
      <EnvWrapper>
        <EnvLabel>Agent Name</EnvLabel>
        <EnvInput
          onChange={(e) => setAgentName(e.targetValue)}
          placeholder="mini-agi"
        />
      </EnvWrapper>
      <EnvWrapper>
        <EnvLabel>Report Location</EnvLabel>
        <EnvInput placeholder="Location from root" />
      </EnvWrapper>
      <EnvWrapper>
        <EnvLabel>OpenAI Key</EnvLabel>
        <EnvInput type="password" placeholder="sk-" />
      </EnvWrapper>
    </CurrentEnvContainer>
  );
};

export default CurrentEnv;

const CurrentEnvContainer = tw.div`
  w-full
  h-full
  flex
  flex-col
  justify-center
`;

const Title = tw.h3`
  font-bold
  text-lg
  text-center
`;

const EnvWrapper = tw.div`
  flex
  mt-4
  justify-between
  items-center
`;

const EnvLabel = tw.label`

`;

const EnvInput = tw.input`
  border
  rounded
  px-2
`;
