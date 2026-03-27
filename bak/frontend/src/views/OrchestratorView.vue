<template>
  <section class="page">
    <h2>任务编排演示</h2>
    <form @submit.prevent="run">
      <div class="grid">
        <label>股票代码<input v-model="ticker" placeholder="如 600519 或 SZ:300750" /></label>
        <label>时间窗口<input v-model="timeWindow" placeholder="如 3M" /></label>
        <label>写作风格<select v-model="style"><option value="professional">专业</option><option value="humor">幽默</option></select></label>
      </div>
      <button type="submit">执行计划</button>
    </form>
    <pre v-if="output">{{ output }}</pre>
  </section>
  
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { api } from '../services/api'

const ticker = ref('')
const timeWindow = ref('3M')
const style = ref<'professional' | 'humor'>('professional')
const output = ref('')

async function run() {
  const plan = [
    { agent: 'DataAgent', params: { ticker: ticker.value, time_window: timeWindow.value } },
    { agent: 'AnalysisAgent', depends_on: ['DataAgent'] },
    { agent: 'WritingAgent', depends_on: ['AnalysisAgent'], params: { style: style.value } },
  ]
  const { data } = await api.post('/plan/run', { plan, context: {} })
  output.value = JSON.stringify(data, null, 2)
}
</script>

<style scoped>
.page { padding: 24px; }
.grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }
pre { background: #111; color: #0f0; padding: 12px; overflow: auto; }
</style>


