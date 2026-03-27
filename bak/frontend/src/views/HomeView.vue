<template>
  <main class="container">
    <h1>AIGC Stocks Agent</h1>
    <p>前后端已初始化。点击下方按钮测试后端健康检查。</p>
    <button @click="checkHealth">检测后端健康</button>
    <p v-if="message">{{ message }}</p>
  </main>
  
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { api } from '../services/api'

const message = ref('')

async function checkHealth() {
  const res = await api.get('/health/')
  message.value = JSON.stringify(res.data)
}
</script>

<style scoped>
.container {
  padding: 24px;
}
button {
  margin-top: 12px;
}
</style>


