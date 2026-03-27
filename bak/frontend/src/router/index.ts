import { createRouter, createWebHistory, RouteRecordRaw } from 'vue-router'

const routes: Array<RouteRecordRaw> = [
  {
    path: '/',
    name: 'Home',
    component: () => import('../views/HomeView.vue'),
  },
  {
    path: '/intent',
    name: 'Intent',
    component: () => import('../views/IntentView.vue'),
  },
  {
    path: '/orchestrator',
    name: 'Orchestrator',
    component: () => import('../views/OrchestratorView.vue'),
  },
  {
    path: '/chart',
    name: 'Chart',
    component: () => import('../views/ChartView.vue'),
  },
  {
    path: '/chat',
    name: 'Chat',
    component: () => import('../views/ChatView.vue'),
  },
]

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes,
})

export default router


