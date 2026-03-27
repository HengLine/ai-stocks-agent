<template>
  <section class="page">
    <h2>智能对话</h2>
    <div class="chat-container">
      <div class="chat-messages" ref="messagesRef">
        <div v-for="msg in messages" :key="msg.id" class="message" :class="msg.role">
          <div class="message-content">{{ msg.content }}</div>
          <div class="message-time">{{ msg.time }}</div>
        </div>
      </div>
      <form @submit.prevent="sendMessage" class="chat-input">
        <input 
          v-model="inputMessage" 
          placeholder="输入您的问题，如：分析宁德时代的走势"
          :disabled="loading"
        />
        <button type="submit" :disabled="loading || !inputMessage.trim()">
          {{ loading ? '发送中...' : '发送' }}
        </button>
      </form>
    </div>
  </section>
</template>

<script setup lang="ts">
import { ref, onMounted, nextTick } from 'vue'
import { api } from '../services/api'

const messages = ref<Array<{id: string, role: string, content: string, time: string}>>([])
const inputMessage = ref('')
const loading = ref(false)
const messagesRef = ref<HTMLDivElement | null>(null)

onMounted(() => {
  loadHistory()
})

async function loadHistory() {
  try {
    const { data } = await api.get('/chat/history', { params: { user_id: 'default_user', limit: 10 } })
    const history = data.history || []
    
    messages.value = history.map((item: any) => ({
      id: item.id,
      role: 'assistant',
      content: item.metadata?.response || '',
      time: new Date().toLocaleTimeString()
    }))
    
    // 添加用户消息
    history.forEach((item: any) => {
      if (item.metadata?.message) {
        messages.value.unshift({
          id: `${item.id}_user`,
          role: 'user',
          content: item.metadata.message,
          time: new Date().toLocaleTimeString()
        })
      }
    })
    
    scrollToBottom()
  } catch (error) {
    console.error('加载历史记录失败:', error)
  }
}

async function sendMessage() {
  if (!inputMessage.value.trim() || loading.value) return
  
  const userMessage = inputMessage.value.trim()
  inputMessage.value = ''
  
  // 添加用户消息
  messages.value.push({
    id: Date.now().toString(),
    role: 'user',
    content: userMessage,
    time: new Date().toLocaleTimeString()
  })
  
  scrollToBottom()
  loading.value = true
  
  try {
    const { data } = await api.post('/chat', {
      message: userMessage,
      user_id: 'default_user'
    })
    
    // 添加助手回复
    messages.value.push({
      id: (Date.now() + 1).toString(),
      role: 'assistant',
      content: data.response,
      time: new Date().toLocaleTimeString()
    })
    
    scrollToBottom()
  } catch (error) {
    console.error('发送消息失败:', error)
    messages.value.push({
      id: (Date.now() + 1).toString(),
      role: 'assistant',
      content: '抱歉，服务暂时不可用，请稍后再试。',
      time: new Date().toLocaleTimeString()
    })
  } finally {
    loading.value = false
  }
}

function scrollToBottom() {
  nextTick(() => {
    if (messagesRef.value) {
      messagesRef.value.scrollTop = messagesRef.value.scrollHeight
    }
  })
}
</script>

<style scoped>
.page { padding: 24px; }
.chat-container { 
  max-width: 800px; 
  margin: 0 auto; 
  border: 1px solid #ddd; 
  border-radius: 8px; 
  overflow: hidden; 
}
.chat-messages { 
  height: 400px; 
  overflow-y: auto; 
  padding: 16px; 
  background: #f9f9f9; 
}
.message { 
  margin-bottom: 16px; 
  display: flex; 
  flex-direction: column; 
}
.message.user { align-items: flex-end; }
.message.assistant { align-items: flex-start; }
.message-content { 
  max-width: 70%; 
  padding: 8px 12px; 
  border-radius: 12px; 
  word-wrap: break-word; 
}
.message.user .message-content { 
  background: #007bff; 
  color: white; 
}
.message.assistant .message-content { 
  background: white; 
  border: 1px solid #ddd; 
}
.message-time { 
  font-size: 12px; 
  color: #666; 
  margin-top: 4px; 
}
.chat-input { 
  display: flex; 
  padding: 16px; 
  background: white; 
  border-top: 1px solid #ddd; 
}
.chat-input input { 
  flex: 1; 
  padding: 8px 12px; 
  border: 1px solid #ddd; 
  border-radius: 4px; 
  margin-right: 8px; 
}
.chat-input button { 
  padding: 8px 16px; 
  background: #007bff; 
  color: white; 
  border: none; 
  border-radius: 4px; 
  cursor: pointer; 
}
.chat-input button:disabled { 
  background: #ccc; 
  cursor: not-allowed; 
}
</style>
