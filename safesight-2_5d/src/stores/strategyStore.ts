import { defineStore } from 'pinia'

export type Strategy = {
  id: string
  name: string
  condition: {
    eventType?: string
    level?: '低' | '中' | '高'
  }
  actions: string[] // e.g., ['弹窗', '短信', '联动门禁']
}

export const useStrategyStore = defineStore('strategy', {
  state: () => ({
    strategies: [] as Strategy[],
  }),
  actions: {
    addStrategy(item: Strategy) {
      this.strategies.push(item)
    },
    removeStrategy(id: string) {
      this.strategies = this.strategies.filter(s => s.id !== id)
    },
  },
})