<template>
  <div class="intent-view">
    <h1>意图识别</h1>
    <div class="intent-form">
      <!-- 输入类型选择 -->
      <div class="form-group">
        <label for="input-type">输入类型</label>
        <select id="input-type" v-model="inputType">
          <option value="auto">自动</option>
          <option value="text">自然语言</option>
          <option value="form">结构化表单</option>
          <option value="voice">语音输入</option>
        </select>
      </div>
      
      <!-- 识别器选择 -->
      <div class="form-group">
        <label for="recognizer">意图识别器</label>
        <select id="recognizer" v-model="recognizer">
          <option value="">默认</option>
          <option value="rule_based">基于规则</option>
          <option value="nlp_based">基于NLP</option>
          <option value="hybrid">混合模式</option>
        </select>
      </div>
      
      <!-- 自然语言输入 -->
      <div v-if="inputType === 'auto' || inputType === 'text'" class="form-group">
        <label for="natural-text">自然语言输入</label>
        <textarea
          id="natural-text"
          v-model="naturalText"
          placeholder="输入分析请求，例如：'帮我分析宁德时代未来3个月走势'"
          rows="4"
        ></textarea>
      </div>
      
      <!-- 结构化表单输入 -->
      <div v-if="inputType === 'auto' || inputType === 'form'">
        <div class="form-row">
          <div class="form-group">
            <label for="ticker">股票代码</label>
            <input
              id="ticker"
              v-model="structuredInput.ticker"
              placeholder="例如：300750"
            />
          </div>
          
          <div class="form-group">
            <label for="time-window">时间窗口</label>
            <input
              id="time-window"
              v-model="structuredInput.timeWindow"
              placeholder="例如：3M"
            />
          </div>
        </div>
        
        <div class="form-row">
          <div class="form-group">
            <label for="dimensions">分析维度</label>
            <select id="dimensions" v-model="structuredInput.dimensions" multiple>
              <option value="technical">技术面</option>
              <option value="fundamental">基本面</option>
              <option value="sentiment">情绪面</option>
            </select>
          </div>
          
          <div class="form-group">
            <label for="output">输出格式</label>
            <select id="output" v-model="structuredInput.output">
              <option value="article">文章</option>
              <option value="video">视频</option>
            </select>
          </div>
        </div>
      </div>
      
      <!-- 语音输入控件 -->
      <div v-if="inputType === 'voice'" class="form-group">
        <label>语音输入</label>
        <div class="voice-controls">
          <button 
            class="voice-button" 
            @click="startRecording"
            :disabled="isRecording"
          >
            开始录音
          </button>
          <button 
            class="voice-button stop" 
            @click="stopRecording"
            :disabled="!isRecording"
          >
            停止录音
          </button>
          <p v-if="recordingTime > 0" class="recording-time">
            录音时长：{{ recordingTime }}秒
          </p>
        </div>
        <p v-if="voiceText" class="voice-text">
          语音转写结果：{{ voiceText }}
        </p>
      </div>
      
      <button class="parse-button" @click="parseIntent">解析意图</button>
    </div>
    
    <div v-if="result" class="result-section">
      <h2>解析结果</h2>
      <pre class="result-code">{{ JSON.stringify(result, null, 2) }}</pre>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onUnmounted } from 'vue'
import api from '@/services/api'

const naturalText = ref('')
const inputType = ref('auto') // auto, text, form, voice
const recognizer = ref('') // rule_based, nlp_based, hybrid
const structuredInput = ref({
  ticker: '',
  timeWindow: '',
  dimensions: [] as string[],
  output: 'article'
})
const result = ref<any>(null)

// 语音相关状态
const isRecording = ref(false)
const recordingTime = ref(0)
const voiceText = ref('')
let recordingInterval: number | null = null
let mediaRecorder: MediaRecorder | null = null
let audioChunks: Blob[] = []

// 开始录音
async function startRecording() {
  try {
    // 检查浏览器支持
    if (!('MediaRecorder' in window)) {
      alert('您的浏览器不支持录音功能')
      return
    }
    
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
    mediaRecorder = new MediaRecorder(stream)
    audioChunks = []
    
    mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        audioChunks.push(event.data)
      }
    }
    
    mediaRecorder.onstop = async () => {
      const audioBlob = new Blob(audioChunks, { type: 'audio/wav' })
      // 这里应该有语音转文本的API调用
      // 为了演示，我们暂时使用模拟数据
      setTimeout(() => {
        voiceText.value = "帮我分析宁德时代未来三个月的技术面走势，生成一篇文章"
        naturalText.value = voiceText.value
      }, 1000)
      
      // 停止所有音轨
      stream.getTracks().forEach(track => track.stop())
    }
    
    mediaRecorder.start()
    isRecording.value = true
    recordingTime.value = 0
    
    // 开始计时
    recordingInterval = window.setInterval(() => {
      recordingTime.value++
    }, 1000)
    
  } catch (error) {
    console.error('录音启动失败:', error)
    alert('录音启动失败，请检查麦克风权限')
    isRecording.value = false
  }
}

// 停止录音
function stopRecording() {
  if (mediaRecorder && isRecording.value) {
    mediaRecorder.stop()
    isRecording.value = false
    
    if (recordingInterval) {
      clearInterval(recordingInterval)
      recordingInterval = null
    }
  }
}

// 解析意图
async function parseIntent() {
  try {
    let payload: any = {
      input_type: inputType.value,
      recognizer: recognizer.value || undefined,
      text: inputType.value === 'voice' ? voiceText.value : naturalText.value,
      ticker: structuredInput.value.ticker,
      time_window: structuredInput.value.timeWindow,
      dimensions: structuredInput.value.dimensions,
      output: structuredInput.value.output
    }
    
    // 添加语音相关信息
    if (inputType.value === 'voice' && recordingTime.value > 0) {
      payload.duration = recordingTime.value
      payload.confidence = 0.9 // 模拟置信度
    }
    
    const response = await api.post('/intent/parse', payload)
    result.value = response.data
  } catch (error) {
    console.error('解析意图失败:', error)
    alert('解析意图失败，请稍后重试')
  }
}

// 组件卸载时清理资源
onUnmounted(() => {
  if (recordingInterval) {
    clearInterval(recordingInterval)
  }
  if (mediaRecorder && isRecording.value) {
    mediaRecorder.stop()
  }
})
</script>

<style scoped>
.intent-view {
  max-width: 800px;
  margin: 0 auto;
  padding: 20px;
}

.intent-form {
  background-color: #f9f9f9;
  padding: 20px;
  border-radius: 8px;
  margin-bottom: 20px;
}

.form-group {
  margin-bottom: 15px;
}

.form-row {
  display: flex;
  gap: 10px;
  margin-bottom: 15px;
}

.form-row .form-group {
  flex: 1;
}

label {
  display: block;
  margin-bottom: 5px;
  font-weight: bold;
}

input, textarea, select {
  width: 100%;
  padding: 8px;
  border: 1px solid #ddd;
  border-radius: 4px;
}

textarea {
  resize: vertical;
}

.parse-button {
  background-color: #4CAF50;
  color: white;
  border: none;
  padding: 10px 20px;
  border-radius: 4px;
  cursor: pointer;
  font-size: 16px;
  width: 100%;
}

.parse-button:hover {
  background-color: #45a049;
}

.result-section {
  background-color: #f0f0f0;
  padding: 20px;
  border-radius: 8px;
}

.result-code {
  background-color: #333;
  color: #f8f8f2;
  padding: 15px;
  border-radius: 4px;
  overflow-x: auto;
  white-space: pre-wrap;
}

/* 语音控制样式 */
.voice-controls {
  display: flex;
  gap: 10px;
  align-items: center;
  margin-bottom: 10px;
}

.voice-button {
  background-color: #2196F3;
  color: white;
  border: none;
  padding: 8px 16px;
  border-radius: 4px;
  cursor: pointer;
}

.voice-button:hover:not(:disabled) {
  background-color: #0b7dda;
}

.voice-button.stop {
  background-color: #f44336;
}

.voice-button.stop:hover:not(:disabled) {
  background-color: #da190b;
}

.voice-button:disabled {
  background-color: #cccccc;
  cursor: not-allowed;
}

.recording-time {
  margin: 0 0 0 10px;
  font-weight: bold;
  color: #f44336;
}

.voice-text {
  background-color: #e3f2fd;
  padding: 10px;
  border-radius: 4px;
  border-left: 4px solid #2196F3;
}
</style>


