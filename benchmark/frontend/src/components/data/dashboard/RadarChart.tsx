import React, { useState } from "react";
import tw from "tailwind-styled-components";

interface RadarChartProps {
  data: any;
}

const RadarChart: React.FC<RadarChartProps> = ({ data }) => {
  return <RadarChartContainer></RadarChartContainer>;
};

export default RadarChart;

const RadarChartContainer = tw.div`
  
`;
