<script setup lang="ts">
import UploadImage from './components/UploadImage.vue'
import DeepZoomViewer from './components/DeepZoomViewer.vue'
import DevicePanel from './components/DevicePanel.vue'
import EventPanel from './components/EventPanel.vue'
import StrategyPanel from './components/StrategyPanel.vue'
import { useAlertStore } from './stores/alertStore'

const alertStore = useAlertStore()

function simulateAlert() {
  alertStore.triggerAlert({
    id: 'evt-' + Date.now(),
    deviceId: 'device-1',
    type: '告警',
    level: '高',
    timestamp: new Date().toISOString(),
    description: '区域入侵预警',
  })
}
</script>

<template>
  <div class="app-container">
    <header class="app-header">
      <h1>2.5D 安检可视化预警系统</h1>
      <div class="header-actions">
        <UploadImage />
        <el-button type="primary" @click="simulateAlert">模拟预警</el-button>
      </div>
    </header>

    <main class="app-main">
      <section class="viewer-section">
        <DeepZoomViewer />
      </section>
      <aside class="side-section">
        <DevicePanel />
        <EventPanel />
        <StrategyPanel />
      </aside>
    </main>
  </div>
</template>

<style scoped>
.app-container {
  display: flex;
  flex-direction: column;
  height: 100vh;
}
.app-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 8px 12px;
  border-bottom: 1px solid #e5e7eb;
}
.header-actions { display: flex; gap: 8px; align-items: center; }
.app-main {
  display: grid;
  grid-template-columns: 1fr 320px;
  gap: 8px;
  flex: 1;
  min-height: 0;
}
.viewer-section {
  position: relative;
  background: #0b0f1a;
}
.side-section {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 8px;
  overflow: auto;
}
</style>
