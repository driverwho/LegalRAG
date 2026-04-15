<template>
  <div class="app-layout">
    <!-- 侧边栏 -->
    <aside class="sidebar">


      <div class="sidebar-content">
        <!-- 会话列表部分 -->
        <div class="section-title session-section-header">
          <span>对话历史</span>
          <el-button 
            type="primary" 
            link 
            size="small" 
            @click="createSession"
            :loading="loadingSessions"
            class="new-session-btn"
          >
            <el-icon><Plus /></el-icon>
            新建对话
          </el-button>
        </div>
        
        <div class="sessions-list" v-loading="loadingSessions">
          <div v-if="sessions.length === 0" class="no-sessions">
            <el-icon><ChatLineRound /></el-icon>
            <span>暂无对话，点击上方新建</span>
          </div>
          <div 
            v-for="session in sessions" 
            :key="session.id" 
            :class="['session-item', { active: activeSessionId === session.id }]"
            @click="selectSession(session.id)"
          >
            <div class="session-info">
              <div class="session-title" :title="session.title">{{ session.title || '未命名对话' }}</div>
              <div class="session-date">{{ formatDate(session.created_at) }}</div>
            </div>
            <el-button
              v-show="activeSessionId === session.id"
              type="danger"
              link
              size="small"
              class="delete-session-btn"
              @click.stop="deleteSession(session.id)"
            >
              <el-icon><Delete /></el-icon>
            </el-button>
          </div>
        </div>

        <!-- 知识库设置部分（可折叠） -->
        <div class="kb-settings-section">
          <div class="kb-settings-header" @click="kbSettingsExpanded = !kbSettingsExpanded">
            <div class="section-title kb-title">知识库设置</div>
            <el-icon :class="['expand-icon', { expanded: kbSettingsExpanded }]"><ArrowRight /></el-icon>
          </div>
          
          <div v-show="kbSettingsExpanded" class="kb-settings-content">
            <div class="upload-box">
              <el-upload
                class="upload-dragger"
                drag
                action="#"
                :auto-upload="false"
                :on-change="handleFileChange"
                :show-file-list="true"
                :multiple="true"
                :limit="1000"
                :on-exceed="handleExceed"
                ref="uploadRef"
              >
                <el-icon class="upload-icon"><UploadFilled /></el-icon>
                <div class="upload-text">
                  <p>点击或拖拽文件至此</p>
                  <span>支持 PDF, DOCX, TXT, CSV</span>
                </div>
              </el-upload>
            </div>

            <el-button 
              type="primary" 
              class="action-btn upload-btn" 
              :loading="uploading" 
              @click="submitUpload"
            >
              {{ uploading ? '正在提交...' : '上传并处理' }}
            </el-button>

            <!-- 任务进度列表 -->
            <div v-if="taskList.length > 0" class="task-list">
              <div class="section-title">处理进度</div>
              <div v-for="task in taskList" :key="task.taskId" class="task-item">
                <div class="task-header">
                  <span class="task-filename">{{ task.fileName }}</span>
                  <div class="task-actions">
                    <span :class="['task-status', task.status.toLowerCase()]">{{ taskStatusLabel(task.status) }}</span>
                    <!-- 取消按钮：只在任务可取消时显示 -->
                    <el-button
                      v-if="canCancelTask(task.status)"
                      type="danger"
                      size="small"
                      :loading="task.cancelling"
                      @click="cancelTask(task.taskId)"
                      class="cancel-btn"
                    >
                      取消
                    </el-button>
                  </div>
                </div>
                <el-progress 
                  :percentage="task.progress || 0" 
                  :status="task.status === 'SUCCESS' ? 'success' : task.status === 'FAILED' || task.status === 'REVOKED' ? 'exception' : ''"
                  :stroke-width="6"
                  class="task-progress"
                />
                <div v-if="task.stage" class="task-stage">{{ task.stage }}</div>
                <div v-if="task.message && (task.status === 'FAILED' || task.status === 'REVOKED')" class="task-error">{{ task.message }}</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div class="sidebar-footer">
        <p>© 2026 Legal RAG System</p>
      </div>
    </aside>

    <!-- 主聊天区域 -->
    <main class="chat-main">
      <!-- 顶部导航 -->
      <header class="chat-header">
        <div class="header-info">
          <h2>智能问答</h2>
        </div>
        <div class="header-actions">
          <el-button circle icon="Delete" @click="clearHistory" title="清空对话" />
        </div>
      </header>

      <!-- 消息列表 -->
      <div class="messages-container" ref="messagesRef" @scroll="onMessagesScroll" @wheel="onMessagesWheel">
        <div v-if="messages.length === 0" class="welcome-screen">
          <div class="welcome-icon">👋</div>
          <h3>你好！我是你的法律知识助手</h3>
          <p>请在左侧上传文档，然后在这里向我提问。</p>
          <div class="feature-list">
            <div class="feature-item">
              <el-icon><Document /></el-icon>
              <span>文档解析</span>
            </div>
            <div class="feature-item">
              <el-icon><Search /></el-icon>
              <span>精准检索</span>
            </div>
            <div class="feature-item">
              <el-icon><ChatLineRound /></el-icon>
              <span>智能问答</span>
            </div>
          </div>
        </div>

        <transition-group name="message-fade">
          <div 
            v-for="(msg, index) in messages" 
            :key="index" 
            :class="['message-row', msg.role]"
          >
            <div class="avatar">
              <el-avatar :size="40" :icon="msg.role === 'user' ? User : Service" :class="msg.role" />
            </div>
            <div class="message-content">
              <!-- 查询纠正提示 -->
              <div v-if="msg.correctedQuery" class="query-correction-bar">
                <el-icon><EditPen /></el-icon>
                <span>已纠正为：<strong>{{ msg.correctedQuery }}</strong></span>
              </div>
              <!-- 查询分类标签 -->
              <div v-if="msg.queryType && msg.queryType !== 'general'" class="query-type-tag">
                <el-tag size="small" :type="msg.queryType === '法条' ? 'primary' : 'warning'" effect="plain" round>
                  {{ msg.queryType === '法条' ? '法条检索' : '案例检索' }}
                </el-tag>
              </div>
              <!-- 进度状态：正在流式接收但内容尚未到达 -->
              <div v-if="msg.streaming && !msg.content" class="bubble status-bubble">
                <el-icon class="spinning-icon"><Loading /></el-icon>
                <span class="status-text">{{ msg.progressText || '处理中...' }}</span>
              </div>
              <!-- 正常内容气泡 -->
              <template v-else>
                <div class="bubble" v-html="renderMarkdown(msg.content)"></div>
                <span v-if="msg.streaming" class="stream-cursor" aria-hidden="true"></span>
              </template>
              
              <!-- 引用来源卡片 -->
              <div v-if="msg.sources && msg.sources.length > 0" class="sources-card">
                <div class="sources-header" @click="toggleSources(index)">
                  <el-icon><CollectionTag /></el-icon>
                  <span>参考依据 ({{ msg.sources.length }})</span>
                  <el-icon :class="['arrow', { rotated: msg.showSources }]"><ArrowDown /></el-icon>
                </div>
                <div v-show="msg.showSources" class="sources-list">
                  <div v-for="(source, sIndex) in msg.sources" :key="sIndex" class="source-item">
                    <div class="source-meta">
                      <span class="index">#{{ sIndex + 1 }}</span>
                      <span class="score">相似度: {{ (source.score * 100).toFixed(1) }}%</span>
                    </div>
                    <div class="source-text">{{ source.content }}</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </transition-group>

        <div v-if="thinking" class="message-row assistant thinking-row">
          <div class="avatar">
            <el-avatar :size="40" :icon="Service" class="assistant" />
          </div>
          <div class="message-content">
            <div class="bubble status-bubble">
              <el-icon class="spinning-icon"><Loading /></el-icon>
              <span class="status-text">连接中...</span>
            </div>
          </div>
        </div>
      </div>

      <!-- 滚到底部按钮 -->
      <transition name="scroll-btn-fade">
        <button
          v-if="userScrolledUp && (streaming || thinking)"
          class="scroll-to-bottom-btn"
          @click="jumpToBottom"
          title="跳到底部"
        >
          <el-icon><ArrowDown /></el-icon>
        </button>
      </transition>

      <!-- 输入区域 -->
      <div class="input-wrapper">
        <div class="input-box">
          <el-input
            v-model="inputQuery"
            placeholder="输入您的问题，按 Enter 发送..."
            type="textarea"
            :autosize="{ minRows: 1, maxRows: 5 }"
            resize="none"
            class="chat-input"
            @keydown.enter.prevent="sendMessage"
          />
          <el-button
            type="primary"
            circle
            class="send-btn"
            :disabled="!inputQuery.trim() || thinking || streaming"
            @click="sendMessage"
          >
            <el-icon><Position /></el-icon>
          </el-button>
        </div>
        <div class="input-footer">
          <span>按 Enter 发送，Shift + Enter 换行</span>
        </div>
      </div>
    </main>
  </div>
</template>

<script setup>
import { ref, nextTick, onMounted, onUnmounted } from 'vue'
import {
  UploadFilled, User, Service, Position, Loading,
  Delete, Document, Search, ChatLineRound,
  CollectionTag, ArrowDown, Plus, ArrowRight, EditPen
} from '@element-plus/icons-vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import axios from 'axios'
import MarkdownIt from 'markdown-it'

const md = new MarkdownIt({ html: true, breaks: true, linkify: true })
// Use relative path so it works in both dev (via Vite proxy) and production (via Nginx proxy)
const API_BASE = '/api/vector'

// State
const COLLECTION_NAME = 'agent_rag'
const uploadRef = ref(null)
const filesToUpload = ref([])
const uploading = ref(false)

// Task tracking
const taskList = ref([])
let pollingTimers = {}

const inputQuery = ref('')
const messages = ref([])
const thinking = ref(false)      // 等待首个 token（显示三点动画）
const streaming = ref(false)     // 正在流式输出中
const messagesRef = ref(null)

// Session management state
const sessions = ref([])
const activeSessionId = ref(null)
const loadingSessions = ref(false)
const kbSettingsExpanded = ref(false)  // 知识库设置展开/折叠
const userScrolledUp = ref(false)      // 用户是否主动向上滚动

// Session message cache: { sessionId: messages[] }
const sessionCache = ref(new Map())

// Markdown 渲染
const renderMarkdown = (text) => {
  return md.render(text || '')
}

// Toggle Sources
const toggleSources = (index) => {
  const msg = messages.value[index]
  msg.showSources = !msg.showSources
}

// 清空历史
const clearHistory = async () => {
  if (messages.value.length === 0) {
    ElMessage.info('当前没有对话内容')
    return
  }

  try {
    await ElMessageBox.confirm('确定要清空当前对话吗？这将删除所有消息记录。', '确认清空', {
      confirmButtonText: '清空',
      cancelButtonText: '取消',
      type: 'warning'
    })

    // 如果有活动会话，删除整个会话并创建新会话
    if (activeSessionId.value) {
      const oldSessionId = activeSessionId.value

      // 删除旧会话
      await axios.delete(`${API_BASE}/sessions/${oldSessionId}`)

      // 从缓存中移除
      sessionCache.value.delete(oldSessionId)

      // 从列表中移除
      const index = sessions.value.findIndex(s => s.id === oldSessionId)
      sessions.value.splice(index, 1)

      // 创建新会话
      const response = await axios.post(`${API_BASE}/sessions`, { title: '新对话' })
      if (response.data.session) {
        const newSession = response.data.session
        sessions.value.unshift(newSession)
        activeSessionId.value = newSession.id
      }
    }

    messages.value = []
    ElMessage.success('对话已清空')
  } catch (error) {
    if (error !== 'cancel') {
      console.error('清空对话失败:', error)
      ElMessage.error('清空对话失败')
    }
  }
}

// File Upload Logic
const handleFileChange = (_file, fileList) => {
  filesToUpload.value = fileList.map(f => f.raw)
}

const handleExceed = () => {
  ElMessage.warning('最多同时上传 10 个文件')
}

const submitUpload = async () => {
  if (filesToUpload.value.length === 0) {
    ElMessage.warning('请先选择文件')
    return
  }

  uploading.value = true

  for (const file of filesToUpload.value) {
    try {
      const formData = new FormData()
      formData.append('file', file)
      formData.append('collection_name', COLLECTION_NAME)

      const response = await axios.post(`${API_BASE}/upload_file`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      })

      if (response.data.success) {
        const task = {
          taskId: response.data.task_id,
          fileName: file.name,
          status: 'PENDING',
          progress: 0,
          stage: null,
          message: response.data.message,
          cancelling: false,
        }
        taskList.value.push(task)
        startPolling(task.taskId)
      }
    } catch (error) {
      console.error(`Upload failed for ${file.name}:`, error)
      taskList.value.push({
        taskId: null,
        fileName: file.name,
        status: 'FAILED',
        progress: 0,
        stage: null,
        message: error.response?.data?.detail || '上传失败',
        cancelling: false,
      })
    }
  }

  uploading.value = false
  uploadRef.value?.clearFiles()
  filesToUpload.value = []
  
  if (taskList.value.some(t => t.status !== 'FAILED')) {
    ElMessage.success(`已提交 ${taskList.value.filter(t => t.status !== 'FAILED').length} 个文件处理任务`)
  }
}

// Chat Logic
const sendMessage = async () => {
  const query = inputQuery.value.trim()
  if (!query || thinking.value || streaming.value) return

  // 如果没有活动会话，先创建一个
  let sessionId = activeSessionId.value
  if (!sessionId) {
    try {
      const response = await axios.post(`${API_BASE}/sessions`, {})
      if (response.data.session) {
        sessionId = response.data.session.id
        sessions.value.unshift(response.data.session)
        activeSessionId.value = sessionId
      }
    } catch (error) {
      console.error('创建会话失败:', error)
      ElMessage.error('创建会话失败，请重试')
      return
    }
  }

  // 锁定当前会话ID，防止流式传输时切换会话导致消息错乱
  const lockedSessionId = sessionId
  const lockedMessages = messages.value

  // 检查是否需要更新会话标题（第一条用户消息）
  const isFirstMessage = messages.value.filter(m => m.role === 'user').length === 0

  messages.value.push({
    role: 'user',
    content: query
  })

  // 立即更新缓存
  if (sessionId) {
    sessionCache.value.set(sessionId, [...messages.value])
  }

  // 如果是第一条消息，更新会话标题
  if (isFirstMessage && sessionId) {
    const title = query.slice(0, 20) + (query.length > 20 ? '...' : '')
    await updateSessionTitle(sessionId, title)
  }

  inputQuery.value = ''
  thinking.value = true   // 显示三点等待动画（等待 HTTP 连接建立）
  scrollToBottom(true)    // 发送时强制到底部，重置滚动状态

  // assistantMsgIndex 在连接成功后再确定
  let assistantMsgIndex = -1

  try {
    const res = await fetch(`${API_BASE}/query/stream/v2`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question: query,
        session_id: lockedSessionId
      })
    })

    if (!res.ok) {
      throw new Error(`HTTP ${res.status}`)
    }

    // HTTP 连接已建立，现在才插入 assistant 气泡并关闭三点动画
    // 这样两者不会同时出现
    thinking.value = false
    streaming.value = true
    assistantMsgIndex = lockedMessages.length
    lockedMessages.push({
      role: 'assistant',
      content: '',
      sources: [],
      showSources: false,
      streaming: true,
      progressText: '',       // 由 progress 事件实时更新
      correctedQuery: null,   // 由 preprocessing_result 事件填充
      queryType: null,        // 查询分类：法条 / 案例 / general
    })

    // 只有当前仍在该会话时才滚动
    if (activeSessionId.value === lockedSessionId) {
      scrollToBottom()
    }

    const reader = res.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })

      // SSE 每条消息以 "\n\n" 结尾
      const parts = buffer.split('\n\n')
      buffer = parts.pop()   // 最后一段可能不完整，留给下次

      for (const part of parts) {
        const line = part.trim()
        if (!line.startsWith('data: ')) continue

        const jsonStr = line.slice('data: '.length)
        let event
        try {
          event = JSON.parse(jsonStr)
        } catch (e) {
          continue
        }

        if (event.type === 'progress') {
          lockedMessages[assistantMsgIndex].progressText = event.text
        } else if (event.type === 'preprocessing_result') {
          // 查询纠正 + 分类结果
          const d = event.data
          lockedMessages[assistantMsgIndex].queryType = d.query_type
          if (d.corrected && d.corrected !== d.original) {
            lockedMessages[assistantMsgIndex].correctedQuery = d.corrected
          }
        } else if (event.type === 'sources') {
          lockedMessages[assistantMsgIndex].sources = event.sources
        } else if (event.type === 'chunk') {
          lockedMessages[assistantMsgIndex].content += event.text
          // 更新缓存中的消息（使用锁定的会话ID）
          sessionCache.value.set(lockedSessionId, [...lockedMessages])
          // 只有当前仍在该会话时才滚动
          if (activeSessionId.value === lockedSessionId) {
            scrollToBottom()
          }
        } else if (event.type === 'done') {
          lockedMessages[assistantMsgIndex].streaming = false
          // 最终更新缓存
          sessionCache.value.set(lockedSessionId, [...lockedMessages])
        } else if (event.type === 'error') {
          lockedMessages[assistantMsgIndex].content = '抱歉，生成回答时出现错误：' + event.message
          lockedMessages[assistantMsgIndex].streaming = false
        }
      }
    }
  } catch (error) {
    console.error('流式请求失败:', error)
    thinking.value = false
    if (assistantMsgIndex >= 0) {
      // 连接建立后才出错：更新已有气泡
      lockedMessages[assistantMsgIndex].content = '网络错误或服务不可用，请检查后端服务是否启动。'
      lockedMessages[assistantMsgIndex].streaming = false
    } else {
      // 连接未建立（fetch 本身失败）：新增错误气泡
      lockedMessages.push({
        role: 'assistant',
        content: '网络错误或服务不可用，请检查后端服务是否启动。',
        sources: [],
        showSources: false,
        streaming: false
      })
    }
    // 更新缓存
    sessionCache.value.set(lockedSessionId, [...lockedMessages])
  } finally {
    thinking.value = false
    streaming.value = false
    // 只有当前仍在该会话时才滚动
    if (activeSessionId.value === lockedSessionId) {
      scrollToBottom()
    }
  }
}

const scrollToBottom = (force = false) => {
  nextTick(() => {
    if (!messagesRef.value) return
    if (force || (!userScrolledUp.value && isAtBottom.value)) {
      messagesRef.value.scrollTop = messagesRef.value.scrollHeight
      if (force) {
        isAtBottom.value = true
        userScrolledUp.value = false
      }
    }
  })
}

// 用户手动滚动：检测是否回到底部（50px 容差）
const isAtBottom = ref(true)
const onMessagesScroll = () => {
  const el = messagesRef.value
  if (!el) return
  const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 50
  isAtBottom.value = atBottom
  if (atBottom) {
    userScrolledUp.value = false  // 用户滚回底部，恢复自动跟随
  }
}

// wheel 向上 → 标记用户主动上划，暂停自动滚动
const onMessagesWheel = (e) => {
  if (e.deltaY < 0) {
    userScrolledUp.value = true
  }
}

// "跳到底部"按钮：强制回到底部并恢复自动跟随
const jumpToBottom = () => {
  userScrolledUp.value = false
  scrollToBottom(true)
}

// Task polling logic
const startPolling = (taskId) => {
  pollingTimers[taskId] = setInterval(async () => {
    try {
      const response = await axios.get(`${API_BASE}/tasks/${taskId}`)
      const data = response.data
      
      const task = taskList.value.find(t => t.taskId === taskId)
      if (!task) {
        stopPolling(taskId)
        return
      }

      task.status = data.status
      task.progress = data.progress || task.progress
      task.stage = data.stage || task.stage
      task.message = data.message || task.message

      if (data.status === 'SUCCESS' || data.status === 'FAILED') {
        stopPolling(taskId)
        if (data.status === 'SUCCESS') {
          ElMessage.success(`${task.fileName} 处理完成！`)
        } else {
          ElMessage.error(`${task.fileName} 处理失败`)
        }
      }
    } catch (error) {
      console.error(`Polling failed for ${taskId}:`, error)
    }
  }, 2000)
}

const stopPolling = (taskId) => {
  if (pollingTimers[taskId]) {
    clearInterval(pollingTimers[taskId])
    delete pollingTimers[taskId]
  }
}

const taskStatusLabel = (status) => {
  const labels = {
    'PENDING': '排队中',
    'STARTED': '处理中',
    'PROGRESS': '处理中',
    'SUCCESS': '已完成',
    'FAILED': '失败',
    'REVOKED': '已取消',
  }
  return labels[status] || status
}

// 判断任务是否可以取消
const canCancelTask = (status) => {
  return status !== 'SUCCESS' && status !== 'FAILED' && status !== 'REVOKED'
}

// 取消任务
const cancelTask = async (taskId) => {
  const task = taskList.value.find(t => t.taskId === taskId)
  if (!task) return

  // 标记为正在取消
  task.cancelling = true

  try {
    const response = await axios.post(`${API_BASE}/tasks/${taskId}/cancel`)
    const data = response.data

    if (data.success) {
      ElMessage.success(data.message)
      // 更新任务状态
      task.status = 'REVOKED'
      task.message = '任务已取消'
      task.progress = 0
      // 停止轮询
      stopPolling(taskId)
    } else {
      ElMessage.info(data.message)
    }
  } catch (error) {
    console.error(`Cancel task failed for ${taskId}:`, error)
    ElMessage.error(error.response?.data?.detail || '取消任务失败')
  } finally {
    task.cancelling = false
  }
}

// ==================== 会话管理功能 ====================

// 格式化日期
const formatDate = (dateStr) => {
  if (!dateStr) return ''
  const date = new Date(dateStr)
  const now = new Date()
  const isToday = date.toDateString() === now.toDateString()

  if (isToday) {
    return date.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })
  }
  return date.toLocaleDateString('zh-CN', { month: 'short', day: 'numeric' })
}

// 加载会话列表
const loadSessions = async () => {
  loadingSessions.value = true
  try {
    const response = await axios.get(`${API_BASE}/sessions`)
    if (response.data.sessions) {
      sessions.value = response.data.sessions
    }
  } catch (error) {
    console.error('加载会话列表失败:', error)
    ElMessage.error('加载会话列表失败')
  } finally {
    loadingSessions.value = false
  }
}

// 创建新会话
const createSession = async () => {
  try {
    const response = await axios.post(`${API_BASE}/sessions`, { title: '新对话' })
    if (response.data.session) {
      const newSession = response.data.session
      sessions.value.unshift(newSession)
      activeSessionId.value = newSession.id
      messages.value = []
      ElMessage.success('创建新对话成功')
    }
  } catch (error) {
    console.error('创建会话失败:', error)
    ElMessage.error('创建会话失败')
  }
}

// 选择会话
const selectSession = async (sessionId) => {
  if (activeSessionId.value === sessionId) return

  // 先保存当前会话的消息到缓存
  if (activeSessionId.value && messages.value.length > 0) {
    sessionCache.value.set(activeSessionId.value, [...messages.value])
  }

  // 检查缓存中是否有该会话的消息
  if (sessionCache.value.has(sessionId)) {
    activeSessionId.value = sessionId
    messages.value = sessionCache.value.get(sessionId)
    return
  }

  // 缓存未命中，从后端加载
  loadingSessions.value = true
  try {
    const response = await axios.get(`${API_BASE}/sessions/${sessionId}`)
    if (response.data.session) {
      activeSessionId.value = sessionId

      // 转换消息格式
      if (response.data.messages) {
        messages.value = response.data.messages.map(msg => {
          let sources = []
          if (msg.sources) {
            if (typeof msg.sources === 'string') {
              try {
                sources = JSON.parse(msg.sources)
              } catch (e) {
                console.warn('解析消息来源失败:', e)
                sources = []
              }
            } else {
              sources = msg.sources
            }
          }
          return {
            role: msg.role,
            content: msg.content,
            sources: sources,
            showSources: false
          }
        })
        // 加载后立即缓存
        sessionCache.value.set(sessionId, [...messages.value])
      } else {
        messages.value = []
      }
    }
  } catch (error) {
    console.error('加载会话失败:', error)
    ElMessage.error('加载会话消息失败')
  } finally {
    loadingSessions.value = false
  }
}

// 删除会话
const deleteSession = async (sessionId) => {
  try {
    await ElMessageBox.confirm('确定要删除这个对话吗？此操作不可恢复。', '确认删除', {
      confirmButtonText: '删除',
      cancelButtonText: '取消',
      type: 'warning'
    })

    await axios.delete(`${API_BASE}/sessions/${sessionId}`)

    // 从缓存中移除
    sessionCache.value.delete(sessionId)

    // 从列表中移除
    const index = sessions.value.findIndex(s => s.id === sessionId)
    sessions.value.splice(index, 1)

    // 如果删除的是当前活动会话，切换到其他会话或清空
    if (activeSessionId.value === sessionId) {
      activeSessionId.value = null
      messages.value = []

      // 自动选择下一个会话
      if (sessions.value.length > 0) {
        await selectSession(sessions.value[0].id)
      }
    }

    ElMessage.success('删除成功')
  } catch (error) {
    if (error !== 'cancel') {
      console.error('删除会话失败:', error)
      ElMessage.error('删除会话失败')
    }
  }
}

// 更新会话标题
const updateSessionTitle = async (sessionId, title) => {
  try {
    await axios.put(`${API_BASE}/sessions/${sessionId}`, { title })
    // 更新本地会话列表中的标题
    const session = sessions.value.find(s => s.id === sessionId)
    if (session) {
      session.title = title
    }
  } catch (error) {
    console.error('更新会话标题失败:', error)
  }
}

// 组件挂载时加载会话列表
onMounted(async () => {
  await loadSessions()

  // 如果有会话，自动选择最新的一个
  if (sessions.value.length > 0) {
    await selectSession(sessions.value[0].id)
  }
})

// 清理 on unmount
onUnmounted(() => {
  Object.keys(pollingTimers).forEach(stopPolling)
})
</script>

<style scoped>
/* 全局布局 */
.app-layout {
  display: flex;
  height: 100%;
  width: 100%;
  background-color: #f0f2f5;
  font-family: 'PingFang SC', 'Microsoft YaHei', sans-serif;
  overflow: hidden;
}

/* 侧边栏样式 */
.sidebar {
  width: 320px;
  background: #1a1c23; /* 深色背景 */
  color: #fff;
  display: flex;
  flex-direction: column;
  box-shadow: 4px 0 15px rgba(0, 0, 0, 0.1);
  z-index: 10;
  transition: all 0.3s ease;
}

.logo-area {
  padding: 30px 20px;
  display: flex;
  align-items: center;
  gap: 12px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}

.logo-icon {
  font-size: 28px;
}

.logo-area h1 {
  font-size: 20px;
  font-weight: 600;
  margin: 0;
  letter-spacing: 1px;
  background: linear-gradient(45deg, #409eff, #36cfc9);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}

.sidebar-content {
  flex: 1;
  padding: 20px;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
}

.section-title {
  font-size: 12px;
  color: rgba(255, 255, 255, 0.4);
  margin-bottom: 15px;
  text-transform: uppercase;
  letter-spacing: 1px;
}

.upload-box {
  margin-bottom: 20px;
}

/* 深度定制 Element Plus 上传组件 */
:deep(.el-upload-dragger) {
  background-color: rgba(255, 255, 255, 0.05) !important;
  border: 1px dashed rgba(255, 255, 255, 0.2) !important;
  border-radius: 12px !important;
  transition: all 0.3s;
}

:deep(.el-upload-dragger:hover) {
  border-color: #409eff !important;
  background-color: rgba(255, 255, 255, 0.08) !important;
}

.upload-icon {
  font-size: 40px;
  color: rgba(255, 255, 255, 0.5);
  margin-bottom: 10px;
}

.upload-text p {
  color: rgba(255, 255, 255, 0.8);
  margin: 0 0 5px 0;
  font-size: 14px;
}

.upload-text span {
  color: rgba(255, 255, 255, 0.4);
  font-size: 12px;
}

.form-group {
  margin-bottom: 20px;
}

.form-group label {
  display: block;
  font-size: 13px;
  color: rgba(255, 255, 255, 0.7);
  margin-bottom: 8px;
}

:deep(.custom-input .el-input__wrapper) {
  background-color: rgba(255, 255, 255, 0.05);
  box-shadow: none;
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 8px;
}

:deep(.custom-input .el-input__inner) {
  color: #fff;
}

.action-btn {
  width: 100%;
  height: 44px;
  font-size: 15px;
  border-radius: 8px;
  font-weight: 500;
  box-shadow: 0 4px 12px rgba(64, 158, 255, 0.3);
}

.success-tip {
  margin-top: 15px;
  padding: 10px;
  background: rgba(103, 194, 58, 0.15);
  border-radius: 8px;
  display: flex;
  align-items: center;
  gap: 8px;
  color: #67c23a;
  font-size: 13px;
}

.sidebar-footer {
  padding: 20px;
  text-align: center;
  border-top: 1px solid rgba(255, 255, 255, 0.1);
  color: rgba(255, 255, 255, 0.3);
  font-size: 12px;
}

/* 主区域样式 */
.chat-main {
  flex: 1;
  display: flex;
  flex-direction: column;
  background: #f5f7fa;
  position: relative;
}

.chat-header {
  height: 70px;
  background: #fff;
  border-bottom: 1px solid #e4e7ed;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 30px;
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.02);
}

.header-info h2 {
  margin: 0;
  font-size: 18px;
  color: #303133;
}

.messages-container {
  flex: 1;
  padding: 30px;
  overflow-y: auto;
}

/* 欢迎屏幕 */
.welcome-screen {
  height: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  color: #909399;
}

.welcome-icon {
  font-size: 60px;
  margin-bottom: 20px;
}

.welcome-screen h3 {
  color: #303133;
  font-size: 24px;
  margin: 0 0 10px 0;
}

.feature-list {
  display: flex;
  gap: 20px;
  margin-top: 40px;
}

.feature-item {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 10px;
  padding: 20px;
  background: #fff;
  border-radius: 12px;
  width: 100px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
  transition: transform 0.3s;
}

.feature-item:hover {
  transform: translateY(-5px);
}

.feature-item .el-icon {
  font-size: 24px;
  color: #409eff;
}

/* 消息气泡 */
.message-row {
  display: flex;
  gap: 15px;
  margin-bottom: 30px;
  max-width: 800px;
  margin-left: auto;
  margin-right: auto;
}

.message-row.user {
  flex-direction: row-reverse;
}

.avatar .el-avatar {
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.avatar .assistant {
  background: linear-gradient(135deg, #36cfc9, #409eff);
}

.avatar .user {
  background: linear-gradient(135deg, #667eea, #764ba2);
}

.message-content {
  max-width: 80%;
}

.bubble {
  padding: 15px 20px;
  border-radius: 16px;
  font-size: 15px;
  line-height: 1.7;
  position: relative;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
}

.user .bubble {
  background: #409eff;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: #fff;
  border-bottom-right-radius: 4px;
}

.assistant .bubble {
  background: #fff;
  color: #303133;
  border-bottom-left-radius: 4px;
}

/* 引用源卡片 */
.sources-card {
  margin-top: 10px;
  background: #fff;
  border-radius: 12px;
  overflow: hidden;
  border: 1px solid #ebeef5;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.02);
}

.sources-header {
  padding: 10px 15px;
  background: #f9fafc;
  display: flex;
  align-items: center;
  gap: 8px;
  cursor: pointer;
  font-size: 13px;
  font-weight: 600;
  color: #606266;
  transition: background 0.2s;
}

.sources-header:hover {
  background: #f0f2f5;
}

.sources-header .arrow {
  margin-left: auto;
  transition: transform 0.3s;
}

.sources-header .arrow.rotated {
  transform: rotate(180deg);
}

.sources-list {
  padding: 0 15px 15px 15px;
}

.source-item {
  margin-top: 12px;
  padding-top: 12px;
  border-top: 1px dashed #ebeef5;
}

.source-item:first-child {
  border-top: none;
}

.source-meta {
  display: flex;
  justify-content: space-between;
  margin-bottom: 4px;
  font-size: 12px;
}

.source-meta .index {
  color: #409eff;
  font-weight: bold;
}

.source-meta .score {
  color: #909399;
}

.source-text {
  font-size: 13px;
  color: #606266;
  line-height: 1.5;
}

/* 输入区域 */
.input-wrapper {
  padding: 20px 30px 30px 30px;
  background: #fff; /* 或保持透明，看设计 */
  background: linear-gradient(to top, #f5f7fa 80%, rgba(245, 247, 250, 0) 100%);
  display: flex;
  flex-direction: column;
  align-items: center;
}

.input-box {
  width: 100%;
  max-width: 800px;
  position: relative;
  background: #fff;
  border-radius: 24px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
  border: 1px solid #ebeef5;
  transition: all 0.3s;
}

.input-box:focus-within {
  box-shadow: 0 4px 24px rgba(64, 158, 255, 0.15);
  border-color: #409eff;
}

:deep(.chat-input .el-textarea__inner) {
  box-shadow: none;
  border: none;
  background: transparent;
  padding: 15px 50px 15px 20px; /* 右侧留出按钮空间 */
  font-size: 15px;
  resize: none;
}

.send-btn {
  position: absolute;
  right: 10px;
  bottom: 10px; /* 或 align with single line */
  width: 36px;
  height: 36px;
}

.input-footer {
  margin-top: 10px;
  font-size: 12px;
  color: #909399;
}

/* 流式输出光标 */
.stream-cursor {
  display: inline-block;
  width: 2px;
  height: 1em;
  background-color: #409eff;
  margin-left: 2px;
  vertical-align: text-bottom;
  animation: blink 0.8s step-start infinite;
}

@keyframes blink {
  0%, 100% { opacity: 1; }
  50%       { opacity: 0; }
}

/* 状态气泡（连接中 / 检索中 / 生成中） */

/* 查询纠正提示条 */
.query-correction-bar {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 6px 12px;
  margin-bottom: 8px;
  background: #ecf5ff;
  border: 1px solid #d9ecff;
  border-radius: 8px;
  font-size: 13px;
  color: #409eff;
  animation: slideDown 0.3s ease;
}

.query-correction-bar .el-icon {
  font-size: 14px;
  flex-shrink: 0;
}

.query-correction-bar strong {
  color: #303133;
  font-weight: 600;
}

/* 查询分类标签 */
.query-type-tag {
  margin-bottom: 6px;
}

.status-bubble {
  display: flex !important;
  align-items: center;
  gap: 10px;
  padding: 12px 18px !important;
  min-width: 180px;
}

.spinning-icon {
  font-size: 18px;
  color: #409eff;
  flex-shrink: 0;
  animation: spin 1s linear infinite;
}

.status-text {
  font-size: 14px;
  color: #909399;
}

@keyframes spin {
  from { transform: rotate(0deg); }
  to   { transform: rotate(360deg); }
}

/* 消息过渡动画 */
.message-fade-enter-active,
.message-fade-leave-active {
  transition: all 0.4s ease;
}

.message-fade-enter-from,
.message-fade-leave-to {
  opacity: 0;
  transform: translateY(20px);
}

/* Markdown 样式适配 */
:deep(.bubble p) {
  margin: 0 0 8px 0;
}

:deep(.bubble p:last-child) {
  margin: 0;
}

:deep(.bubble ul), :deep(.bubble ol) {
  padding-left: 20px;
  margin: 8px 0;
}

:deep(.bubble code) {
  background: rgba(0, 0, 0, 0.1);
  padding: 2px 4px;
  border-radius: 4px;
  font-family: monospace;
}

.user :deep(.bubble code) {
  background: rgba(255, 255, 255, 0.2);
}

/* Task list styles */
.task-list {
  margin-top: 20px;
}

.task-item {
  padding: 12px;
  background: rgba(255, 255, 255, 0.05);
  border-radius: 8px;
  margin-bottom: 10px;
  border: 1px solid rgba(255, 255, 255, 0.08);
}

.task-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
}

.task-filename {
  font-size: 13px;
  color: rgba(255, 255, 255, 0.85);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  max-width: 160px;
}

.task-status {
  font-size: 11px;
  padding: 2px 8px;
  border-radius: 10px;
  font-weight: 500;
}

.task-status.pending { background: rgba(144, 147, 153, 0.2); color: #909399; }
.task-status.started,
.task-status.progress { background: rgba(64, 158, 255, 0.2); color: #409eff; }
.task-status.success { background: rgba(103, 194, 58, 0.2); color: #67c23a; }
.task-status.failed { background: rgba(245, 108, 108, 0.2); color: #f56c6c; }
.task-status.revoked { background: rgba(230, 162, 60, 0.2); color: #e6a23c; }

.task-actions {
  display: flex;
  align-items: center;
  gap: 8px;
}

.cancel-btn {
  padding: 2px 8px !important;
  font-size: 11px !important;
  height: 22px !important;
}

.task-stage {
  font-size: 11px;
  color: rgba(255, 255, 255, 0.4);
  margin-top: 4px;
}

.task-error {
  font-size: 11px;
  color: #f56c6c;
  margin-top: 4px;
}

:deep(.task-progress .el-progress-bar__outer) {
  background-color: rgba(255, 255, 255, 0.1) !important;
}

:deep(.task-progress .el-progress__text) {
  color: rgba(255, 255, 255, 0.6) !important;
  font-size: 11px !important;
}

/* 会话列表样式 */
.session-section-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}

.new-session-btn {
  color: #409eff;
  font-size: 13px;
}

.new-session-btn:hover {
  color: #66b1ff;
}

.sessions-list {
  flex: 1;
  overflow-y: auto;
  margin-bottom: 20px;
  min-height: 100px;
}

.no-sessions {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 40px 20px;
  color: rgba(255, 255, 255, 0.4);
  font-size: 13px;
  text-align: center;
}

.no-sessions .el-icon {
  font-size: 32px;
  margin-bottom: 10px;
  color: rgba(255, 255, 255, 0.2);
}

.session-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 14px;
  margin-bottom: 8px;
  background: rgba(255, 255, 255, 0.05);
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.25s ease;
  border-left: 3px solid transparent;
}

.session-item:hover {
  background: rgba(255, 255, 255, 0.08);
}

.session-item.active {
  background: rgba(255, 255, 255, 0.1);
  border-left: 3px solid #409eff;
}

.session-info {
  flex: 1;
  min-width: 0;
}

.session-title {
  font-size: 14px;
  color: rgba(255, 255, 255, 0.85);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  margin-bottom: 4px;
  font-weight: 500;
}

.session-date {
  font-size: 12px;
  color: rgba(255, 255, 255, 0.4);
}

.delete-session-btn {
  opacity: 0;
  color: rgba(255, 255, 255, 0.4);
  padding: 4px !important;
  transition: all 0.2s ease;
}

.session-item:hover .delete-session-btn,
.session-item.active .delete-session-btn {
  opacity: 1;
}

.delete-session-btn:hover {
  color: #f56c6c;
}

/* 知识库设置区域 */
.kb-settings-section {
  border-top: 1px solid rgba(255, 255, 255, 0.1);
  padding-top: 15px;
}

.kb-settings-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  cursor: pointer;
  padding: 5px 0;
  transition: all 0.2s ease;
}

.kb-settings-header:hover .kb-title {
  color: rgba(255, 255, 255, 0.7);
}

.kb-title {
  margin-bottom: 0 !important;
}

.expand-icon {
  font-size: 14px;
  color: rgba(255, 255, 255, 0.4);
  transition: transform 0.3s ease;
}

.expand-icon.expanded {
  transform: rotate(90deg);
}

.kb-settings-content {
  padding-top: 15px;
  animation: slideDown 0.3s ease;
}

@keyframes slideDown {
  from {
    opacity: 0;
    transform: translateY(-10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

/* 欢迎屏幕调整 */
:deep(.el-loading-mask) {
  background-color: rgba(26, 28, 35, 0.8);
}

/* 跳到底部按钮 */
.scroll-to-bottom-btn {
  position: absolute;
  bottom: 130px;
  left: 50%;
  transform: translateX(-50%);
  width: 36px;
  height: 36px;
  border-radius: 50%;
  border: none;
  background: #409eff;
  color: #fff;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 4px 12px rgba(64, 158, 255, 0.4);
  z-index: 10;
  transition: background 0.2s, transform 0.2s;
}

.scroll-to-bottom-btn:hover {
  background: #66b1ff;
  transform: translateX(-50%) scale(1.08);
}

.scroll-btn-fade-enter-active,
.scroll-btn-fade-leave-active {
  transition: opacity 0.25s, transform 0.25s;
}

.scroll-btn-fade-enter-from,
.scroll-btn-fade-leave-to {
  opacity: 0;
  transform: translateX(-50%) translateY(8px);
}
</style>
