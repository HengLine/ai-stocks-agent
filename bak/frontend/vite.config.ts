import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// https://vite.dev/config/
export default defineConfig({
  plugins: [vue()],
  server: {
    port: 8080,  // 使用1024以上的端口
    host: '0.0.0.0'  // 允许外部访问
  }
})
