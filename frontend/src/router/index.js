import { createRouter, createWebHistory } from 'vue-router'
import RagChat from '@/components/RagChat.vue'
import KnowledgeBase from '@/components/KnowledgeBase.vue'

const routes = [
  { path: '/', name: 'chat', component: RagChat },
  { path: '/knowledge', name: 'knowledge', component: KnowledgeBase },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

export default router
