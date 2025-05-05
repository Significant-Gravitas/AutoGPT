import { APIKeysSection } from "@/components/agptui/composite/APIKeySection";

const ApiKeysPage = () => {
  return (
    <main className="flex-1 space-y-7.5 pb-8">
      <h1 className="font-poppins text-[1.75rem] font-medium leading-[2.5rem] text-zinc-500">
        API key
      </h1>
      <APIKeysSection />
    </main>
  );
};

export default ApiKeysPage;
