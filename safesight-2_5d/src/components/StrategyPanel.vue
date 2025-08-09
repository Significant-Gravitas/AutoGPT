<template>
  <el-card shadow="never" header="联动策略">
    <el-form :inline="true" @submit.prevent>
      <el-form-item label="名称">
        <el-input v-model="form.name" placeholder="如 入侵-高等级联动" style="width: 160px" />
      </el-form-item>
      <el-form-item label="类型">
        <el-input v-model="form.eventType" placeholder="告警/烟雾/入侵" style="width: 120px" />
      </el-form-item>
      <el-form-item label="等级">
        <el-select v-model="form.level" placeholder="选择" style="width: 100px">
          <el-option label="低" value="低" />
          <el-option label="中" value="中" />
          <el-option label="高" value="高" />
        </el-select>
      </el-form-item>
      <el-form-item label="动作">
        <el-select v-model="form.actions" multiple placeholder="选择" style="width: 200px">
          <el-option label="弹窗" value="弹窗" />
          <el-option label="短信" value="短信" />
          <el-option label="联动门禁" value="联动门禁" />
        </el-select>
      </el-form-item>
      <el-form-item>
        <el-button type="primary" :disabled="!form.name" @click="add">添加策略</el-button>
      </el-form-item>
    </el-form>

    <el-table :data="strategyStore.strategies" size="small" style="width: 100%; margin-top: 8px">
      <el-table-column prop="name" label="名称" width="180" />
      <el-table-column label="条件">
        <template #default="{ row }">{{ row.condition.eventType || '任意' }} / {{ row.condition.level || '任意' }}</template>
      </el-table-column>
      <el-table-column prop="actions" label="动作" width="220">
        <template #default="{ row }">{{ row.actions.join('、') }}</template>
      </el-table-column>
      <el-table-column label="操作" width="100">
        <template #default="{ row }">
          <el-popconfirm title="删除该策略？" @confirm="remove(row.id)">
            <template #reference>
              <el-button size="small" type="danger">删除</el-button>
            </template>
          </el-popconfirm>
        </template>
      </el-table-column>
    </el-table>
  </el-card>
</template>

<script setup lang="ts">
import { reactive } from 'vue'
import { useStrategyStore } from '../stores/strategyStore'

const strategyStore = useStrategyStore()
const form = reactive({ name: '', eventType: '', level: '' as '低' | '中' | '高' | '', actions: [] as string[] })

function add() {
  strategyStore.addStrategy({
    id: 'stg-' + Date.now(),
    name: form.name,
    condition: { eventType: form.eventType || undefined, level: (form.level as any) || undefined },
    actions: form.actions.slice(),
  })
  form.name = ''
  form.eventType = ''
  form.level = ''
  form.actions = []
}
function remove(id: string) {
  strategyStore.removeStrategy(id)
}
</script>