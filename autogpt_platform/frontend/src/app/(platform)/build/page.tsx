import { Metadata } from "next";
import MainBuilderPage from "./components/MainBuilderPage";
import { getV1GetSpecificGraph } from "@/app/api/__generated__/endpoints/graphs/graphs";

export async function generateMetadata({
  searchParams,
}: {
  searchParams: Promise<{ flowID: string; flowVersion: string }>;
}): Promise<Metadata> {
  const { flowID, flowVersion } = await searchParams;

  if (!flowID || !flowVersion) {
    return {
      title: `Builder - AutoGPT Platform`,
    };
  }

  const { data: graph } = await getV1GetSpecificGraph(flowID, {
    version: parseInt(flowVersion),
  });

  if (!graph || typeof graph !== "object" || !("name" in graph)) {
    return {
      title: `Builder - AutoGPT Platform`,
    };
  }

  return {
    title: `${graph.name} - Builder - AutoGPT Platform`,
  };
}
const BuilderPage = () => {
  return <MainBuilderPage />;
};

export default BuilderPage;
