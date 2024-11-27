"use client";

import * as React from "react";
import {
    Popover,
    PopoverTrigger,
    PopoverContent,
    PopoverAnchor,
} from "@/components/ui/popover";
import { PublishAgentSelect, Agent } from "../PublishAgentSelect";
import { PublishAgentInfo } from "../PublishAgentSelectInfo";
import { PublishAgentAwaitingReview } from "../PublishAgentAwaitingReview";
import { Button } from "../Button";
import { StoreSubmissionRequest, MyAgentsResponse } from "@/lib/autogpt-server-api";
import { createClient } from "@/lib/supabase/client";
import AutoGPTServerAPI from "@/lib/autogpt-server-api/client";
import { useRouter } from "next/navigation";
interface PublishAgentPopoutProps {
    trigger?: React.ReactNode;
    agents: {
        name: string;
        lastEdited: string;
        imageSrc: string;
    }[];
    onOpenBuilder?: () => void;
}

export const PublishAgentPopout: React.FC<PublishAgentPopoutProps> = ({
    trigger,
    agents,
    onOpenBuilder,
}) => {
    const [step, setStep] = React.useState<"select" | "info" | "review">("select");
    const [myAgents, setMyAgents] = React.useState<MyAgentsResponse | null>(null);
    const [selectedAgent, setSelectedAgent] = React.useState<string | null>(null);
    const [publishData, setPublishData] = React.useState<StoreSubmissionRequest>({
        name: "",
        sub_heading: "",
        slug: "",
        description: "",
        image_urls: [],
        agent_id: "",
        agent_version: 0,
        categories: [],
    });
    const [selectedAgentId, setSelectedAgentId] = React.useState<string | null>(null);
    const [selectedAgentVersion, setSelectedAgentVersion] = React.useState<number | null>(null);
    const [open, setOpen] = React.useState(false);

    const popupId = React.useId();
    const router = useRouter();

    const supabase = createClient();

    const api = new AutoGPTServerAPI(
        process.env.NEXT_PUBLIC_AGPT_SERVER_URL,
        process.env.NEXT_PUBLIC_AGPT_WS_SERVER_URL,
        supabase,
    );


    React.useEffect(() => {
        const loadMyAgents = async () => {
            try {
                const response = await api.getMyAgents();
                setMyAgents(response);
            } catch (error) {
                console.error("Failed to load my agents:", error);
            }
        };

        loadMyAgents();
    }, []);


    const handleClose = () => {
        setStep("select");
        setSelectedAgent(null);
        setPublishData({
            name: "",
            sub_heading: "",
            description: "",
            image_urls: [],
            agent_id: "",
            agent_version: 0,
            slug: "",
            categories: [],
        });
        setOpen(false);
    };

    const handleAgentSelect = (agentName: string) => {
        setSelectedAgent(agentName);
    };

    const handleNextFromSelect = (agentId: string, agentVersion: number) => {
        setStep("info");
        setSelectedAgentId(agentId);
        setSelectedAgentVersion(agentVersion);
    };

    const handleNextFromInfo = async (
        name: string,
        subHeading: string,
        slug: string,
        description: string,
        imageUrls: string[],
        videoUrl: string,
        categories: string[]
    ) => {

        if (!name || !subHeading || !description || !imageUrls.length || !categories.length) {
            console.error("Missing required fields");
            return;
        }

        setPublishData({
            name,
            sub_heading: subHeading,
            description,
            image_urls: imageUrls,
            video_url: videoUrl,
            agent_id: selectedAgentId || "",
            agent_version: selectedAgentVersion || 0,
            slug,
            categories
        });

        // Create store submission
        try {
            const submission = await api.createStoreSubmission({
                name: name,
                sub_heading: subHeading,
                description: description,
                image_urls: imageUrls,
                video_url: videoUrl,
                agent_id: selectedAgentId || "",
                agent_version: selectedAgentVersion || 0,
                slug: slug.replace(/\s+/g, '-'),
                categories: categories
            });
            console.log("Store submission created:", submission);
        } catch (error) {
            console.error("Error creating store submission:", error);
        }
        setStep("review");
    };


    const handleBack = () => {
        if (step === "info") {
            setStep("select");
        } else if (step === "review") {
            setStep("info");
        }
    };

    const renderContent = () => {
        switch (step) {
            case "select":
                return (
                    <div className="flex min-h-screen items-center justify-center">
                        <div className="mx-auto flex w-full max-w-[900px] flex-col rounded-3xl bg-white shadow-lg">
                            <div className="h-full overflow-y-auto">
                                <PublishAgentSelect
                                    agents={myAgents?.agents.map(agent => ({
                                        name: agent.agent_name,
                                        id: agent.agent_id,
                                        version: agent.agent_version,
                                        lastEdited: agent.last_edited,
                                        imageSrc: "https://picsum.photos/300/200" // Fallback image if none provided
                                    })) || []}
                                    onSelect={handleAgentSelect}
                                    onCancel={handleClose}
                                    onNext={handleNextFromSelect}
                                    onClose={handleClose}
                                    onOpenBuilder={onOpenBuilder || (() => router.push('/build'))}
                                />
                            </div>
                        </div>
                    </div>
                );
            case "info":
                return (
                    <div className="flex min-h-screen items-center justify-center">
                        <div className="mx-auto flex w-full max-w-[900px] flex-col rounded-3xl bg-white shadow-lg">
                            <div className="h-[700px] overflow-y-auto">
                                <PublishAgentInfo
                                    onBack={handleBack}
                                    onSubmit={handleNextFromInfo}
                                    onClose={handleClose}
                                />
                            </div>
                        </div>
                    </div>
                );
            case "review":
                return publishData ? (
                    <div className="flex justify-center">
                        <div className="mx-auto flex w-full max-w-[900px] flex-col rounded-3xl bg-white shadow-lg">
                            <div className="h-[600px] overflow-y-auto">
                                <PublishAgentAwaitingReview
                                    agentName={publishData.name}
                                    subheader={publishData.sub_heading}
                                    description={publishData.description}
                                    thumbnailSrc={publishData.image_urls[0]}
                                    onClose={handleClose}
                                    onDone={handleClose}
                                    onViewProgress={() => router.push('/store/dashboard')}
                                />
                            </div>
                        </div>
                    </div>
                ) : null;
        }
    };

    return (
        <Popover open={open} onOpenChange={setOpen}>
            <PopoverTrigger asChild>
                {trigger || (
                    <Button variant="default">
                        Publish Agent
                    </Button>
                )}
            </PopoverTrigger>
            <PopoverAnchor asChild>
                <div className="hidden fixed top-0 left-0 h-screen w-screen items-center justify-center">
                </div>
            </PopoverAnchor>

            <PopoverContent
                id={popupId}
                align="center"
                className="w-screen h-screen bg-transparent z-50"
            >
            {renderContent()}
            </PopoverContent>
        </Popover>
    );
};
