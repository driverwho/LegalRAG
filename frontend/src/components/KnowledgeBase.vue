<template>
  <div class="kb-layout">
    <!-- 工具栏 -->
    <header class="kb-toolbar">
      <div class="toolbar-left">
        <el-select
          v-model="selectedCollection"
          placeholder="选择集合"
          class="collection-select"
          @change="onCollectionChange"
          :loading="loadingCollections"
        >
          <el-option
            v-for="col in collections"
            :key="col.name"
            :label="`${col.name} (${col.document_count})`"
            :value="col.name"
          />
        </el-select>

        <el-input
          v-model="keyword"
          placeholder="搜索文档内容..."
          clearable
          class="search-input"
          @keydown.enter="loadDocuments"
          @clear="loadDocuments"
        >
          <template #prefix>
            <el-icon><Search /></el-icon>
          </template>
        </el-input>

        <el-button type="primary" @click="loadDocuments" :loading="loadingDocs">
          <el-icon><Search /></el-icon>
          搜索
        </el-button>

        <el-button @click="refresh" :loading="loadingDocs">
          <el-icon><Refresh /></el-icon>
          刷新
        </el-button>
      </div>

      <div class="toolbar-right">
        <el-button
          type="danger"
          :disabled="selectedIds.length === 0"
          @click="batchDelete"
        >
          <el-icon><Delete /></el-icon>
          批量删除 ({{ selectedIds.length }})
        </el-button>
      </div>
    </header>

    <!-- 文档表格 -->
    <div class="kb-table-wrapper">
      <el-table
        ref="tableRef"
        :data="documents"
        v-loading="loadingDocs"
        stripe
        border
        style="width: 100%"
        height="100%"
        @selection-change="onSelectionChange"
        row-key="id"
      >
        <el-table-column type="selection" width="50" reserve-selection />
        <el-table-column label="ID" prop="id" width="220" show-overflow-tooltip />
        <el-table-column label="内容预览" min-width="400">
          <template #default="{ row }">
            <span class="content-preview">{{ truncate(row.content, 200) }}</span>
          </template>
        </el-table-column>
        <el-table-column label="来源" width="200" show-overflow-tooltip>
          <template #default="{ row }">
            {{ row.metadata?.source || '-' }}
          </template>
        </el-table-column>
        <el-table-column label="元数据" width="120" align="center">
          <template #default="{ row }">
            <el-tag size="small" type="info">{{ Object.keys(row.metadata || {}).length }} 项</el-tag>
          </template>
        </el-table-column>
        <el-table-column label="操作" width="200" align="center" fixed="right">
          <template #default="{ row }">
            <el-button type="primary" link size="small" @click="viewDocument(row)">
              <el-icon><View /></el-icon> 详情
            </el-button>
            <el-button type="warning" link size="small" @click="editDocument(row)">
              <el-icon><Edit /></el-icon> 编辑
            </el-button>
            <el-button type="danger" link size="small" @click="deleteSingle(row.id)">
              <el-icon><Delete /></el-icon> 删除
            </el-button>
          </template>
        </el-table-column>
      </el-table>
    </div>

    <!-- 分页 -->
    <div class="kb-pagination">
      <el-pagination
        v-model:current-page="currentPage"
        v-model:page-size="pageSize"
        :total="total"
        :page-sizes="[10, 20, 50, 100]"
        layout="total, sizes, prev, pager, next, jumper"
        @current-change="onPageChange"
        @size-change="onSizeChange"
      />
    </div>

    <!-- 详情弹窗 -->
    <el-dialog
      v-model="detailVisible"
      :title="isEditing ? '编辑文档' : '文档详情'"
      width="750px"
      destroy-on-close
    >
      <div v-if="currentDoc" class="doc-detail">
        <div class="detail-field">
          <label>文档 ID</label>
          <el-input :model-value="currentDoc.id" disabled />
        </div>

        <div class="detail-field">
          <label>内容</label>
          <el-input
            v-model="editForm.content"
            type="textarea"
            :autosize="{ minRows: 6, maxRows: 20 }"
            :disabled="!isEditing"
          />
        </div>

        <div class="detail-field">
          <label>元数据</label>
          <div v-if="!isEditing" class="metadata-view">
            <div v-for="(value, key) in currentDoc.metadata" :key="key" class="meta-row">
              <span class="meta-key">{{ key }}</span>
              <span class="meta-value">{{ value }}</span>
            </div>
            <div v-if="Object.keys(currentDoc.metadata || {}).length === 0" class="meta-empty">
              无元数据
            </div>
          </div>
          <div v-else class="metadata-edit">
            <div v-for="(value, key) in editForm.metadata" :key="key" class="meta-edit-row">
              <el-input :model-value="key" disabled class="meta-key-input" />
              <el-input v-model="editForm.metadata[key]" class="meta-value-input" />
              <el-button type="danger" link @click="removeMetaField(key)">
                <el-icon><Delete /></el-icon>
              </el-button>
            </div>
            <div class="meta-add-row">
              <el-input v-model="newMetaKey" placeholder="键名" class="meta-key-input" />
              <el-input v-model="newMetaValue" placeholder="值" class="meta-value-input" />
              <el-button type="primary" link @click="addMetaField" :disabled="!newMetaKey.trim()">
                <el-icon><Plus /></el-icon>
              </el-button>
            </div>
          </div>
        </div>
      </div>

      <template #footer>
        <div class="dialog-footer">
          <el-button @click="detailVisible = false">
            {{ isEditing ? '取消' : '关闭' }}
          </el-button>
          <el-button v-if="!isEditing" type="primary" @click="startEdit">
            <el-icon><Edit /></el-icon> 编辑
          </el-button>
          <el-button v-else type="primary" :loading="saving" @click="saveEdit">
            <el-icon><Check /></el-icon> 保存
          </el-button>
        </div>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted } from 'vue'
import { Search, Refresh, Delete, View, Edit, Check, Plus } from '@element-plus/icons-vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import axios from 'axios'

// Use relative path so it works in both dev (via Vite proxy) and production (via Nginx proxy)
const API_BASE = '/api/vector'

// State
const collections = ref([])
const selectedCollection = ref('')
const loadingCollections = ref(false)

const documents = ref([])
const loadingDocs = ref(false)
const keyword = ref('')
const total = ref(0)
const currentPage = ref(1)
const pageSize = ref(20)

const selectedIds = ref([])
const tableRef = ref(null)

// Detail / Edit dialog
const detailVisible = ref(false)
const isEditing = ref(false)
const saving = ref(false)
const currentDoc = ref(null)
const editForm = reactive({ content: '', metadata: {} })
const newMetaKey = ref('')
const newMetaValue = ref('')

// Helpers
const truncate = (text, len) => {
  if (!text) return ''
  return text.length > len ? text.slice(0, len) + '...' : text
}

// Load collections
const loadCollections = async () => {
  loadingCollections.value = true
  try {
    const res = await axios.get(`${API_BASE}/collections`)
    collections.value = res.data.collections || []
    if (collections.value.length > 0 && !selectedCollection.value) {
      selectedCollection.value = collections.value[0].name
      await loadDocuments()
    }
  } catch (e) {
    console.error('加载集合失败:', e)
    ElMessage.error('加载集合列表失败')
  } finally {
    loadingCollections.value = false
  }
}

// Load documents
const loadDocuments = async () => {
  if (!selectedCollection.value) return
  loadingDocs.value = true
  try {
    const offset = (currentPage.value - 1) * pageSize.value
    const params = { offset, limit: pageSize.value }
    if (keyword.value.trim()) params.keyword = keyword.value.trim()

    const res = await axios.get(
      `${API_BASE}/knowledge/${selectedCollection.value}/documents`,
      { params }
    )
    documents.value = res.data.documents || []
    total.value = res.data.total || 0
  } catch (e) {
    console.error('加载文档失败:', e)
    ElMessage.error('加载文档列表失败')
  } finally {
    loadingDocs.value = false
  }
}

const onCollectionChange = () => {
  currentPage.value = 1
  keyword.value = ''
  selectedIds.value = []
  loadDocuments()
}

const onPageChange = () => loadDocuments()
const onSizeChange = () => {
  currentPage.value = 1
  loadDocuments()
}

const refresh = () => {
  keyword.value = ''
  currentPage.value = 1
  loadCollections()
}

const onSelectionChange = (rows) => {
  selectedIds.value = rows.map(r => r.id)
}

// View detail
const viewDocument = (row) => {
  currentDoc.value = { ...row }
  editForm.content = row.content
  editForm.metadata = { ...(row.metadata || {}) }
  isEditing.value = false
  detailVisible.value = true
}

// Edit
const editDocument = (row) => {
  currentDoc.value = { ...row }
  editForm.content = row.content
  editForm.metadata = { ...(row.metadata || {}) }
  isEditing.value = true
  detailVisible.value = true
}

const startEdit = () => {
  isEditing.value = true
}

const removeMetaField = (key) => {
  delete editForm.metadata[key]
}

const addMetaField = () => {
  const k = newMetaKey.value.trim()
  if (!k) return
  editForm.metadata[k] = newMetaValue.value
  newMetaKey.value = ''
  newMetaValue.value = ''
}

const saveEdit = async () => {
  if (!currentDoc.value) return
  saving.value = true
  try {
    const body = {}
    if (editForm.content !== currentDoc.value.content) {
      body.content = editForm.content
    }
    // Always send metadata when editing (could have added/removed keys)
    body.metadata = { ...editForm.metadata }
    // Convert all metadata values to strings for ChromaDB compatibility
    for (const key of Object.keys(body.metadata)) {
      if (body.metadata[key] !== null && body.metadata[key] !== undefined) {
        body.metadata[key] = String(body.metadata[key])
      }
    }

    const res = await axios.put(
      `${API_BASE}/knowledge/${selectedCollection.value}/documents/${currentDoc.value.id}`,
      body
    )
    if (res.data.success) {
      ElMessage.success('更新成功')
      detailVisible.value = false
      loadDocuments()
      loadCollections()
    }
  } catch (e) {
    console.error('更新文档失败:', e)
    ElMessage.error(e.response?.data?.detail || '更新失败')
  } finally {
    saving.value = false
  }
}

// Delete
const deleteSingle = async (id) => {
  try {
    await ElMessageBox.confirm('确定要删除此文档吗？此操作不可恢复。', '确认删除', {
      confirmButtonText: '删除',
      cancelButtonText: '取消',
      type: 'warning',
    })
    const res = await axios.delete(
      `${API_BASE}/knowledge/${selectedCollection.value}/documents`,
      { data: { ids: [id] } }
    )
    if (res.data.success) {
      ElMessage.success('删除成功')
      loadDocuments()
      loadCollections()
    }
  } catch (e) {
    if (e !== 'cancel') {
      console.error('删除失败:', e)
      ElMessage.error('删除失败')
    }
  }
}

const batchDelete = async () => {
  if (selectedIds.value.length === 0) return
  try {
    await ElMessageBox.confirm(
      `确定要删除选中的 ${selectedIds.value.length} 个文档吗？此操作不可恢复。`,
      '批量删除',
      { confirmButtonText: '删除', cancelButtonText: '取消', type: 'warning' }
    )
    const res = await axios.delete(
      `${API_BASE}/knowledge/${selectedCollection.value}/documents`,
      { data: { ids: selectedIds.value } }
    )
    if (res.data.success) {
      ElMessage.success(res.data.message)
      selectedIds.value = []
      loadDocuments()
      loadCollections()
    }
  } catch (e) {
    if (e !== 'cancel') {
      console.error('批量删除失败:', e)
      ElMessage.error('批量删除失败')
    }
  }
}

onMounted(() => {
  loadCollections()
})
</script>

<style scoped>
.kb-layout {
  display: flex;
  flex-direction: column;
  height: 100%;
  background: #f5f7fa;
}

.kb-toolbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px 24px;
  background: #fff;
  border-bottom: 1px solid #e4e7ed;
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.02);
  flex-shrink: 0;
}

.toolbar-left {
  display: flex;
  align-items: center;
  gap: 12px;
}

.collection-select {
  width: 260px;
}

.search-input {
  width: 280px;
}

.kb-table-wrapper {
  flex: 1;
  padding: 16px 24px 0;
  overflow: hidden;
}

.content-preview {
  font-size: 13px;
  color: #606266;
  line-height: 1.5;
}

.kb-pagination {
  display: flex;
  justify-content: center;
  padding: 16px 24px;
  background: #fff;
  border-top: 1px solid #e4e7ed;
  flex-shrink: 0;
}

/* Detail dialog */
.doc-detail {
  max-height: 60vh;
  overflow-y: auto;
}

.detail-field {
  margin-bottom: 20px;
}

.detail-field label {
  display: block;
  font-size: 13px;
  font-weight: 600;
  color: #303133;
  margin-bottom: 8px;
}

.metadata-view {
  background: #f9fafc;
  border-radius: 8px;
  padding: 12px 16px;
  border: 1px solid #ebeef5;
}

.meta-row {
  display: flex;
  align-items: baseline;
  padding: 6px 0;
  border-bottom: 1px dashed #ebeef5;
}

.meta-row:last-child {
  border-bottom: none;
}

.meta-key {
  font-weight: 600;
  color: #409eff;
  min-width: 120px;
  font-size: 13px;
}

.meta-value {
  color: #606266;
  font-size: 13px;
  word-break: break-all;
}

.meta-empty {
  color: #909399;
  font-size: 13px;
  text-align: center;
  padding: 10px;
}

.metadata-edit {
  background: #f9fafc;
  border-radius: 8px;
  padding: 12px 16px;
  border: 1px solid #ebeef5;
}

.meta-edit-row,
.meta-add-row {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 8px;
}

.meta-key-input {
  width: 160px;
  flex-shrink: 0;
}

.meta-value-input {
  flex: 1;
}

.dialog-footer {
  display: flex;
  justify-content: flex-end;
  gap: 10px;
}
</style>
