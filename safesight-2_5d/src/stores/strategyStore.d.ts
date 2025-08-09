export type Strategy = {
    id: string;
    name: string;
    condition: {
        eventType?: string;
        level?: '低' | '中' | '高';
    };
    actions: string[];
};
export declare const useStrategyStore: import("pinia").StoreDefinition<"strategy", {
    strategies: Strategy[];
}, {}, {
    addStrategy(item: Strategy): void;
    removeStrategy(id: string): void;
}>;
