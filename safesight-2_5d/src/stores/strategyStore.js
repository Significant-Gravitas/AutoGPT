import { defineStore } from 'pinia';
export const useStrategyStore = defineStore('strategy', {
    state: () => ({
        strategies: [],
    }),
    actions: {
        addStrategy(item) {
            this.strategies.push(item);
        },
        removeStrategy(id) {
            this.strategies = this.strategies.filter(s => s.id !== id);
        },
    },
});
