import React, { useState } from "react";
import tw from "tailwind-styled-components";

import RadarChart from "./dashboard/RadarChart";
import CategorySuccess from "./dashboard/CategorySuccess";
import CurrentEnv from "./dashboard/CurrentEnv";

interface DashboardProps {
  data: any;
}

const Dashboard: React.FC<DashboardProps> = ({ data }) => {
  return (
    <DashboardContainer>
      <CardWrapper>
        <RadarChart />
      </CardWrapper>
      <CardWrapper>
        <CategorySuccess />
      </CardWrapper>
      <CardWrapper>
        <CurrentEnv />
      </CardWrapper>
    </DashboardContainer>
  );
};

export default Dashboard;

const DashboardContainer = tw.div`
  w-full
  h-96
  flex
  justify-between
  items-center
`;

const CardWrapper = tw.div`
  w-[30%]
  h-72
  rounded-xl
  shadow-lg
  border
  p-4
`;
