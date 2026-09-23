import { AutopilotAvatar } from "@/components/molecules/AutopilotAvatar/AutopilotAvatar";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";

export function TeamScene() {
  return (
    <div
      className="flex h-full flex-col items-center justify-center gap-8"
      aria-label="Otto and your team of AI Experts"
    >
      <div className="flex flex-col items-center gap-2">
        <AutopilotAvatar size={104} />
        <span>Otto · Your personal Head of AI</span>
      </div>
      <div className="flex gap-10">
        <div className="flex flex-col items-center gap-2">
          <ExpertAvatar
            name="Maria"
            avatarUrl="/autogpt-characters/v1.1/expert-maria/neutral/128.webp"
            size={96}
          />
          <span>Maria · SEO Content Manager</span>
        </div>
        <div className="flex flex-col items-center gap-2">
          <ExpertAvatar
            name="Mina"
            avatarUrl="/autogpt-characters/v1.1/expert-mina/neutral/128.webp"
            size={96}
          />
          <span>Mina · Bookkeeper</span>
        </div>
      </div>
    </div>
  );
}
