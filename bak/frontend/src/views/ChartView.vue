<template>
  <section class="page">
    <h2>K线与指标</h2>
    <form @submit.prevent="load">
      <label>股票代码<input v-model="ticker" placeholder="如 SZ:300750 或 600519" /></label>
      <label>时间窗口<input v-model="window" placeholder="如 3M/6M/1Y" /></label>
      <button type="submit">加载数据</button>
    </form>
    <div class="tools">
      <label><input type="checkbox" v-model="showMA" /> 显示MA(5/10)</label>
      <label><input type="checkbox" v-model="showMACD" /> 显示MACD</label>
      <label><input type="checkbox" v-model="showRSI" /> 显示RSI</label>
    </div>
    <div ref="chartRef" class="chart"></div>
  </section>
  
</template>

<script setup lang="ts">
import { ref, onMounted, watch } from 'vue'
import * as echarts from 'echarts'
import { api } from '../services/api'

const ticker = ref('')
const window = ref('3M')
const chartRef = ref<HTMLDivElement | null>(null)
let chart: echarts.ECharts | null = null
const showMA = ref(true)
const showMACD = ref(false)
const showRSI = ref(false)
let lastData: any = null

onMounted(() => {
  if (chartRef.value) {
    chart = echarts.init(chartRef.value)
  }
})

async function load() {
  const { data } = await api.get('/market/kline', { params: { ticker: ticker.value, window: window.value } })
  const klines = data.klines || []
  const dates = klines.map((k: any) => k.date)
  const closes = klines.map((k: any) => k.close)
  const inds = data.indicators || {}
  lastData = { dates, klines, inds }
  render()
}

watch([showMA, showMACD, showRSI], () => render())

function render() {
  if (!chart || !lastData) return
  const { dates, klines, inds } = lastData
  const candle = klines.map((k: any) => [k.open, k.close, k.low, k.high])

  const grids = [
    { left: 50, right: 20, top: 40, height: 260 }, // 主图
  ] as any[]
  const xAxes: any[] = [{ type: 'category', data: dates, gridIndex: 0 }]
  const yAxes: any[] = [{ type: 'value', scale: true, gridIndex: 0 }]
  const series: echarts.SeriesOption[] = [
    { name: 'K线', type: 'candlestick', data: candle, xAxisIndex: 0, yAxisIndex: 0 },
  ]

  if (showMA.value) {
    if (inds.ma5) series.push({ name: 'MA5', type: 'line', data: inds.ma5, xAxisIndex: 0, yAxisIndex: 0, smooth: true })
    if (inds.ma10) series.push({ name: 'MA10', type: 'line', data: inds.ma10, xAxisIndex: 0, yAxisIndex: 0, smooth: true })
  }

  let gridIdx = 0
  if (showMACD.value) {
    gridIdx += 1
    grids.push({ left: 50, right: 20, top: grids[0].top + grids[0].height + 40, height: 140 })
    xAxes.push({ type: 'category', data: dates, gridIndex: gridIdx, axisLabel: { show: false } })
    yAxes.push({ type: 'value', scale: true, gridIndex: gridIdx })
    if (inds.macd) series.push({ name: 'MACD', type: 'bar', data: inds.macd, xAxisIndex: gridIdx, yAxisIndex: gridIdx })
    if (inds.dif) series.push({ name: 'DIF', type: 'line', data: inds.dif, xAxisIndex: gridIdx, yAxisIndex: gridIdx, smooth: true })
    if (inds.dea) series.push({ name: 'DEA', type: 'line', data: inds.dea, xAxisIndex: gridIdx, yAxisIndex: gridIdx, smooth: true })
  }

  if (showRSI.value) {
    gridIdx += 1
    const top = grids[grids.length - 1].top + grids[grids.length - 1].height + 40
    grids.push({ left: 50, right: 20, top, height: 120 })
    xAxes.push({ type: 'category', data: dates, gridIndex: gridIdx, axisLabel: { show: true } })
    yAxes.push({ type: 'value', scale: true, gridIndex: gridIdx })
    if (inds.rsi) series.push({ name: 'RSI', type: 'line', data: inds.rsi, xAxisIndex: gridIdx, yAxisIndex: gridIdx, smooth: true })
  }

  const option: echarts.EChartsOption = {
    tooltip: { trigger: 'axis' },
    legend: { top: 0 },
    grid: grids,
    xAxis: xAxes,
    yAxis: yAxes,
    dataZoom: [
      { type: 'inside', xAxisIndex: xAxes.map((_, i) => i) },
      { type: 'slider', xAxisIndex: [0], top: 20 },
    ],
    series,
  }
  chart.setOption(option, true)
}
</script>

<style scoped>
.page { padding: 24px; }
.chart { width: 100%; height: 420px; margin-top: 12px; }
form { display: flex; gap: 12px; align-items: center; }
input { padding: 6px; }
</style>


