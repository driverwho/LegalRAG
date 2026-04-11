<script setup>
import { ref, computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ChatLineRound, Folder } from '@element-plus/icons-vue'

const route = useRoute()
const router = useRouter()

const activeMenu = computed(() => route.path)

const handleMenuSelect = (path) => {
  router.push(path)
}
</script>

<template>
  <div class="app-root">
    <nav class="top-nav">
      <div class="nav-brand">
        <span class="brand-icon">🧠</span>
        <span class="brand-text">法律RAG智能助手</span>
      </div>
      <el-menu
        :default-active="activeMenu"
        mode="horizontal"
        class="nav-menu"
        :ellipsis="false"
        @select="handleMenuSelect"
      >
        <el-menu-item index="/">
          <el-icon><ChatLineRound /></el-icon>
          <span>智能问答</span>
        </el-menu-item>
        <el-menu-item index="/knowledge">
          <el-icon><Folder /></el-icon>
          <span>知识库管理</span>
        </el-menu-item>
      </el-menu>
    </nav>
    <div class="app-body">
      <router-view />
    </div>
  </div>
</template>

<style>
body {
  margin: 0;
  padding: 0;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
}

.app-root {
  display: flex;
  flex-direction: column;
  height: 100vh;
  width: 100vw;
  overflow: hidden;
}

.top-nav {
  height: 56px;
  background: #1a1c23;
  display: flex;
  align-items: center;
  padding: 0 24px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
  z-index: 100;
  flex-shrink: 0;
}

.nav-brand {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-right: 40px;
}

.brand-icon {
  font-size: 24px;
}

.brand-text {
  font-size: 18px;
  font-weight: 600;
  background: linear-gradient(45deg, #409eff, #36cfc9);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  white-space: nowrap;
}

.nav-menu {
  background: transparent !important;
  border-bottom: none !important;
  height: 56px;
}

.nav-menu .el-menu-item {
  color: rgba(255, 255, 255, 0.65) !important;
  border-bottom: 2px solid transparent !important;
  height: 56px;
  line-height: 56px;
  font-size: 14px;
}

.nav-menu .el-menu-item:hover {
  color: #fff !important;
  background-color: rgba(255, 255, 255, 0.08) !important;
}

.nav-menu .el-menu-item.is-active {
  color: #fff !important;
  border-bottom: 2px solid #409eff !important;
  background-color: transparent !important;
}

.app-body {
  flex: 1;
  overflow: hidden;
}
</style>
