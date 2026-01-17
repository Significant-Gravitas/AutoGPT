import { withRoleAccess } from "@/lib/withRoleAccess";
import { GenerateTestDataButton } from "./components/GenerateTestDataButton";
import { Text } from "@/components/atoms/Text/Text";

function TestDataDashboard() {
  return (
    <div className="mx-auto p-6">
      <div className="flex flex-col gap-6">
        <div className="flex items-center justify-between">
          <div>
            <Text variant="h1" className="text-3xl">
              Test Data Generation
            </Text>
            <Text variant="body" className="text-gray-500">
              Generate sample data for testing and development
            </Text>
          </div>
        </div>

        <div className="rounded-lg border bg-white p-6 shadow-sm">
          <Text variant="h2" className="mb-4 text-xl">
            Generate Test Data
          </Text>
          <Text variant="body" className="mb-6 text-gray-600">
            Use this tool to populate the database with sample test data. This
            is useful for development and testing purposes.
          </Text>

          <div className="mb-6">
            <Text variant="body-medium" className="mb-2">
              Available Script Types:
            </Text>
            <ul className="list-inside list-disc space-y-2 text-gray-600">
              <li>
                <Text variant="body" as="span">
                  <Text variant="body-medium" as="span">
                    E2E Test Data:
                  </Text>{" "}
                  Creates 15 test users with graphs, library agents, presets,
                  store submissions, and API keys. Uses API functions for better
                  compatibility.
                </Text>
              </li>
              <li>
                <Text variant="body" as="span">
                  <Text variant="body-medium" as="span">
                    Full Test Data:
                  </Text>{" "}
                  Creates 100+ users with comprehensive test data including
                  agent blocks, nodes, executions, analytics, and more. Takes
                  longer to complete.
                </Text>
              </li>
            </ul>
          </div>

          <GenerateTestDataButton />
        </div>

        <div className="rounded-lg border bg-gray-50 p-6">
          <Text variant="body-medium" className="mb-2 text-gray-700">
            What data is created?
          </Text>
          <div className="grid gap-4 text-sm text-gray-600 md:grid-cols-2">
            <div>
              <Text variant="body-medium">E2E Script:</Text>
              <ul className="mt-1 list-inside list-disc">
                <li>
                  <Text variant="small" as="span">
                    15 test users
                  </Text>
                </li>
                <li>
                  <Text variant="small" as="span">
                    15 graphs per user
                  </Text>
                </li>
                <li>
                  <Text variant="small" as="span">
                    Library agents
                  </Text>
                </li>
                <li>
                  <Text variant="small" as="span">
                    Agent presets
                  </Text>
                </li>
                <li>
                  <Text variant="small" as="span">
                    Store submissions
                  </Text>
                </li>
                <li>
                  <Text variant="small" as="span">
                    API keys
                  </Text>
                </li>
                <li>
                  <Text variant="small" as="span">
                    Creator profiles
                  </Text>
                </li>
              </ul>
            </div>
            <div>
              <Text variant="body-medium">Full Script:</Text>
              <ul className="mt-1 list-inside list-disc">
                <li>
                  <Text variant="small" as="span">
                    100 users
                  </Text>
                </li>
                <li>
                  <Text variant="small" as="span">
                    100 agent blocks
                  </Text>
                </li>
                <li>
                  <Text variant="small" as="span">
                    Multiple graphs per user
                  </Text>
                </li>
                <li>
                  <Text variant="small" as="span">
                    Agent nodes and links
                  </Text>
                </li>
                <li>
                  <Text variant="small" as="span">
                    Graph executions
                  </Text>
                </li>
                <li>
                  <Text variant="small" as="span">
                    Store listings and reviews
                  </Text>
                </li>
                <li>
                  <Text variant="small" as="span">
                    Analytics data
                  </Text>
                </li>
                <li>
                  <Text variant="small" as="span">
                    Credit transactions
                  </Text>
                </li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default async function TestDataDashboardPage() {
  "use server";
  const withAdminAccess = await withRoleAccess(["admin"]);
  const ProtectedTestDataDashboard = await withAdminAccess(TestDataDashboard);
  return <ProtectedTestDataDashboard />;
}
