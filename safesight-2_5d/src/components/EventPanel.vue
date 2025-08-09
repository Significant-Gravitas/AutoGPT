<template>
  <el-card shadow="never" header="事件预警">
    <el-table :data="alertStore.events" size="small" style="width: 100%" @row-click="onRowClick">
      <el-table-column prop="timestamp" label="时间" width="170" />
      <el-table-column prop="deviceId" label="设备ID" width="120" />
      <el-table-column prop="type" label="类型" width="100" />
      <el-table-column prop="level" label="等级" width="80" />
      <el-table-column label="描述">
        <template #default="{ row }">{{ row.description }}</template>
      </el-table-column>
      <el-table-column label="状态" width="80">
        <template #default="{ row }">
          <el-tag :type="row.resolved ? 'info' : 'danger'">{{ row.resolved ? '已处理' : '未处理' }}</el-tag>
        </template>
      </el-table-column>
      <el-table-column label="操作" width="220">
        <template #default="{ row }">
          <el-button size="small" @click.stop="focusDevice(row.deviceId)">定位</el-button>
          <el-button size="small" @click.stop="showDetails(row)">详情</el-button>
          <el-button size="small" type="success" :disabled="row.resolved" @click.stop="resolve(row.id)">处理</el-button>
        </template>
      </el-table-column>
    </el-table>

    <el-dialog v-model="detailVisible" title="事件详情" width="460px">
      <el-descriptions :column="1" border>
        <el-descriptions-item label="事件ID">{{ detail?.id }}</el-descriptions-item>
        <el-descriptions-item label="设备ID">{{ detail?.deviceId }}</el-descriptions-item>
        <el-descriptions-item label="类型">{{ detail?.type }}</el-descriptions-item>
        <el-descriptions-item label="等级">{{ detail?.level }}</el-descriptions-item>
        <el-descriptions-item label="时间">{{ detail?.timestamp }}</el-descriptions-item>
        <el-descriptions-item label="描述">{{ detail?.description || '-' }}</el-descriptions-item>
        <el-descriptions-item label="状态">{{ detail?.resolved ? '已处理' : '未处理' }}</el-descriptions-item>
      </el-descriptions>
      <template #footer>
        <el-button @click="detailVisible=false">关闭</el-button>
        <el-button type="success" :disabled="detail?.resolved" @click="detail && resolve(detail.id)">标记处理</el-button>
      </template>
    </el-dialog>
  </el-card>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { useAlertStore, type AlertEvent } from '../stores/alertStore'
import { useImageStore } from '../stores/imageStore'

const alertStore = useAlertStore()
const imageStore = useImageStore()

const detailVisible = ref(false)
const detail = ref<AlertEvent | null>(null)

function resolve(id: string) {
  alertStore.resolveAlert(id)
  if (detail.value?.id === id) detail.value.resolved = true
}
function focusDevice(deviceId: string) {
  imageStore.selectMarker(deviceId)
}
function showDetails(row: AlertEvent) {
  detail.value = row
  detailVisible.value = true
}
function onRowClick(row: AlertEvent) {
  showDetails(row)
}
</script>