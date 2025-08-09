<template>
  <el-card shadow="never" header="设备管理">
    <el-form :inline="true" @submit.prevent>
      <el-form-item label="设备ID">
        <el-input v-model="form.id" placeholder="如 device-1" style="width: 140px" />
      </el-form-item>
      <el-form-item label="名称">
        <el-input v-model="form.name" placeholder="摄像头A" style="width: 140px" />
      </el-form-item>
      <el-form-item>
        <el-button type="primary" :disabled="!form.id || !form.name" @click="startPlace">在图上添加</el-button>
        <el-button @click="cancelPlace" :disabled="!imageStore.pendingMarker">取消</el-button>
      </el-form-item>
    </el-form>

    <el-table :data="imageStore.markers" size="small" style="width: 100%; margin-top: 8px">
      <el-table-column prop="id" label="ID" width="120" />
      <el-table-column prop="name" label="名称" />
      <el-table-column label="坐标" width="140">
        <template #default="{ row }">
          {{ row.x.toFixed(3) }}, {{ row.y.toFixed(3) }}
        </template>
      </el-table-column>
      <el-table-column label="操作" width="160">
        <template #default="{ row }">
          <el-button size="small" @click="select(row.id)">定位</el-button>
          <el-popconfirm title="删除该设备？" @confirm="remove(row.id)">
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
import { useImageStore } from '../stores/imageStore'

const imageStore = useImageStore()
const form = reactive({ id: '', name: '' })

function startPlace() {
  imageStore.startPlacingMarker(form.id.trim(), form.name.trim())
}
function cancelPlace() {
  imageStore.cancelPlacingMarker()
}
function select(id: string) {
  imageStore.selectMarker(id)
}
function remove(id: string) {
  imageStore.removeMarker(id)
}
</script>