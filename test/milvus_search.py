"""
用途：在 Milvus 知识库上执行向量检索与 RAG 生成的示例/测试。
特性：支持 HyDE 查询增强与重排序（Cohere Rerank），便于效果评估与调试。
"""
import logging
import os
from typing import List, Dict, Optional, Tuple, Any, Union
from concurrent.futures import ThreadPoolExecutor

from pymilvus import connections, Collection, utility
from fastapi import HTTPException

from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models import ChatTongyi
from langchain_community.llms.tongyi import Tongyi
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
"""
RAG优化方案参考：https://mp.weixin.qq.com/s/oOja2wXB-gUFt1X_xddtTw
https://dashboard.cohere.com/api-keys
"""

class KnowledgeBaseSearcher:
    def __init__(self, reranker_enabled: bool = True, hyde_enabled: bool = True):
        """
        初始化 KnowledgeBaseSearcher。

        Args:
            reranker_enabled: 是否启用重排序功能
            hyde_enabled: 是否启用HyDE查询增强
        """
        logger.info("Initializing KnowledgeBaseSearcher...")

        # 配置参数
        self.reranker_enabled = reranker_enabled
        self.hyde_enabled = hyde_enabled
        self.default_top_k = 5  # 增大默认检索数量以提高召回率
        self.max_workers = 4  # 并行处理的最大工作线程数

        # 初始化主要聊天模型
        self.chat_model = ChatTongyi()
        logger.info("Initialized ChatTongyi model.")

        # 初始化备用模型（用于故障恢复）
        try:
            self.backup_model = Tongyi(model_name="qwen-max")
            logger.info("Initialized backup model: Tongyi Qwen-max.")
        except Exception as e:
            logger.warning(f"Failed to initialize backup model: {str(e)}")
            self.backup_model = None

        # 初始化嵌入模型
        self.embeddings_model = DashScopeEmbeddings(
            model="text-embedding-v1",
            dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
        )
        logger.info("Initialized DashScope embeddings model.")

        # 初始化重排序器(如果启用)
        if self.reranker_enabled :
            try:
                self.reranker = CohereRerank(
                    cohere_api_key="ECh2AlpW8MhIx5Ip1rEvAKaAeV5kVAAGgMD3Dvx7",
                    top_n=3  # 重排序后保留的文档数量
                )
                logger.info("Initialized Cohere reranker.")
            except Exception as e:
                logger.warning(f"Failed to initialize reranker: {str(e)}")
                self.reranker = None
                self.reranker_enabled = False
        else:
            self.reranker = None
            if self.reranker_enabled:
                logger.warning("Reranker is enabled but COHERE_API_KEY is not set.")
                self.reranker_enabled = False

        # 连接到 Milvus 数据库
        try:
            host = os.getenv("MILVUS_HOST")
            port = os.getenv("MILVUS_PORT")
            connections.connect("default", host=host, port=port)
            logger.info(f"Connected to Milvus database at {host}:{port}.")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to connect to Milvus: {str(e)}")

    def get_query_embedding(self, query_text: str) -> List[float]:
        """
        获取查询文本的 embedding 向量。

        Args:
            query_text: 查询文本

        Returns:
            嵌入向量
        """
        logger.info(f"Generating embedding for query: {query_text}")
        try:
            embedding = self.embeddings_model.embed_documents([query_text])[0]
            logger.info("Embedding generated successfully.")
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error generating embedding: {str(e)}")

    def generate_hypothetical_document(self, query_text: str) -> str:
        """
        实现HyDE (Hypothetical Document Embeddings)方法，生成假设性文档来增强查询效果。

        Args:
            query_text: 用户的原始查询

        Returns:
            生成的假设性文档
        """
        if not self.hyde_enabled:
            return query_text

        logger.info(f"Generating hypothetical document for query: {query_text}")
        try:
            hyde_prompt = PromptTemplate(
                input_variables=["question"],
                template="""基于以下问题，生成一个假设的文档段落，该文档段落可能包含问题的答案。
                问题: {question}
                假设文档:"""
            )

            hypothetical_doc = self.chat_model.invoke(
                hyde_prompt.format(question=query_text)
            ).content

            logger.info(f"Generated hypothetical document: {hypothetical_doc[:100]}...")
            return hypothetical_doc
        except Exception as e:
            logger.warning(f"Error generating hypothetical document: {str(e)}. Falling back to original query.")
            return query_text

    def search_similar_text(
            self,
            user_id: str,
            query_text: str,
            top_k: int = None,
            knowledge_names: Optional[Union[str, List[str]]] = None
    ) -> List[Dict[str, Any]]:
        """
        执行向量检索，根据用户 ID 检索用户知识库中的相似内容。

        Args:
            user_id: 用户ID
            query_text: 查询文本
            top_k: 返回的最大结果数
            knowledge_names: 知识库名称列表或逗号分隔的字符串

        Returns:
            相似文本列表
        """
        if top_k is None:
            top_k = self.default_top_k

        collection_name = f"collection_{int(user_id) % 10}"
        logger.info(f"Searching in collection: {collection_name} for user: {user_id}")

        # 检查集合是否存在
        if not utility.has_collection(collection_name):
            logger.warning(f"Collection {collection_name} does not exist for user: {user_id}")
            raise HTTPException(status_code=404, detail=f"No knowledge base found for user: {user_id}")

        # 获取用户的集合
        try:
            collection = Collection(name=collection_name)
            collection.load()  # 确保集合已加载到内存
            logger.info(f"Collection {collection_name} loaded. Proceeding with search.")
        except Exception as e:
            logger.error(f"Error loading collection: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error loading collection: {str(e)}")

        # 如果启用了HyDE，先生成假设性文档
        if self.hyde_enabled:
            hypothetical_doc = self.generate_hypothetical_document(query_text)
            query_embedding = self.get_query_embedding(hypothetical_doc)
        else:
            # 获取查询文本的向量表示
            query_embedding = self.get_query_embedding(query_text)

        # 设置检索参数
        search_params = {
            "metric_type": "L2",  # 使用 L2 距离进行相似度计算
            "params": {"nprobe": 10}  # 设置搜索精度
        }

        # 构造 `expr` 条件
        expr = self._build_expression(user_id, knowledge_names)

        # 执行向量搜索
        try:
            logger.info(f"Performing vector search with top_k={top_k}...")
            results = collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=["name", "text", "user_id"],
                expr=expr
            )
            logger.info(f"Vector search completed. Found {len(results[0])} results.")
        except Exception as e:
            logger.error(f"Error during vector search: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error during vector search: {str(e)}")

        # 提取相似文本
        search_results = []
        for hits in results:
            for hit in hits:
                result = {
                    "source": hit.entity.get('name'),
                    "text": hit.entity.get('text'),
                    "score": hit.score,
                    "user_id": hit.entity.get('user_id')
                }
                search_results.append(result)
                logger.info(f"Found match: {result['source']} with score: {result['score']}")

        # 如果启用了重排序且有足够的结果，使用Cohere进行重排序
        if self.reranker_enabled and self.reranker and len(search_results) > 1:
            try:
                logger.info("Applying Cohere reranking...")
                docs_for_rerank = [{"id": i, "text": item["text"]} for i, item in enumerate(search_results)]
                reranked_results = self.reranker.rerank(docs_for_rerank, query_text)

                # 重新排序搜索结果
                reranked_indices = [doc["id"] for doc in reranked_results]
                search_results = [search_results[i] for i in reranked_indices]
                logger.info("Reranking completed.")
            except Exception as e:
                logger.warning(f"Error during reranking: {str(e)}. Using original ranking.")

        if not search_results:
            logger.warning("No matching results found.")
            return []

        return search_results

    def _build_expression(self, user_id: str, knowledge_names: Optional[Union[str, List[str]]]) -> Optional[str]:
        """
        构建Milvus查询表达式。

        Args:
            user_id: 用户ID
            knowledge_names: 知识库名称列表或逗号分隔的字符串

        Returns:
            Milvus查询表达式
        """
        # 如何输入 "," 就是搜索知识库的全部内容
        if not knowledge_names:
            return None

        # 确保 knowledge_names 是字符串列表
        if isinstance(knowledge_names, str):
            file_name_list = [name.strip() for name in knowledge_names.split(",") if name.strip()]
        else:
            file_name_list = [name.strip() for name in knowledge_names if name.strip()]

        if not file_name_list:
            logger.warning("No valid knowledge names provided.")
            return None

        if len(file_name_list) > 1:
            # 使用 IN 操作符构造表达式，并将 user_id 追加到每个文件名
            formatted_names = "', '".join([f"{name}_{user_id}" for name in file_name_list])
            expr = f"name in ['{formatted_names}']"
        else:
            # 单个文件名使用 == 并将 user_id 追加
            expr = f"name == '{file_name_list[0]}_{user_id}'"

        logger.info(f"Milvus Search Condition: Applying filter condition: {expr}")
        return expr

    def generate_response(self, contexts: List[Dict[str, Any]], question: str) -> str:
        """
        基于上下文列表生成回答。

        Args:
            contexts: 检索到的上下文列表
            question: 用户问题

        Returns:
            生成的回答
        """
        logger.info("Generating response based on retrieved contexts and user question.")

        # 优化后的prompt模板，更好地处理缺失内容和格式要求
        prompt_template = """
            你是一个专业的知识库助手。请根据下面提供的上下文信息，回答用户的问题。

            上下文信息:
            {context}

            用户问题: {question}

            请遵循以下规则：
            1. 仅基于提供的上下文信息回答问题，不要编造不存在的信息
            2. 如果上下文中没有足够的信息回答问题，请清楚地表明"基于提供的信息，我无法完全回答这个问题"，并提供上下文中可能相关的部分信息
            3. 如果上下文包含表格或结构化数据，请保持其格式清晰
            4. 提供具体、全面的回答，而不是过于笼统的信息
            5. 回答应该清晰、准确、有条理

            回答:
        """

        # 准备上下文
        if not contexts:
            return "未找到与问题相关的信息。请尝试重新表述您的问题或提供更多细节。"

        # 重排序上下文，将最相关的内容放在前面
        if len(contexts) > 1:
            # 根据相似度分数排序（分数越低越相似）
            contexts = sorted(contexts, key=lambda x: x.get("score", float('inf')))

        # 合并上下文
        context_texts = [f"来源: {ctx['source']}\n内容: {ctx['text']}" for ctx in contexts]
        combined_context = "\n\n---\n\n".join(context_texts)

        try:
            # 使用主模型生成回答
            prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)
            formatted_prompt = prompt.format(context=combined_context, question=question)
            response = self.chat_model.invoke(formatted_prompt)
            logger.info("Response generated successfully using primary model.")
            return response.content
        except Exception as e:
            logger.error(f"Error generating response with primary model: {str(e)}")

            # 如果有备用模型，尝试使用备用模型
            if self.backup_model:
                try:
                    logger.info("Attempting to use backup model...")
                    response = self.backup_model.invoke(formatted_prompt)
                    logger.info("Response generated successfully using backup model.")
                    return response
                except Exception as backup_error:
                    logger.error(f"Error generating response with backup model: {str(backup_error)}")

            # 如果都失败了，返回错误信息
            raise HTTPException(status_code=500, detail="Failed to generate response. Please try again later.")

    def parallel_search(self, user_id: str, query_text: str, knowledge_names_list: List[str]) -> List[Dict[str, Any]]:
        """
        并行搜索多个知识库。

        Args:
            user_id: 用户ID
            query_text: 查询文本
            knowledge_names_list: 知识库名称列表

        Returns:
            所有相似文本的合并结果
        """
        logger.info(f"Performing parallel search across {len(knowledge_names_list)} knowledge bases")

        all_results = []

        # 定义单个知识库搜索函数
        def search_single_kb(kb_name):
            try:
                return self.search_similar_text(
                    user_id=user_id,
                    query_text=query_text,
                    top_k=2,  # 每个知识库取少量结果
                    knowledge_names=kb_name
                )
            except Exception as e:
                logger.error(f"Error searching knowledge base {kb_name}: {str(e)}")
                return []

        # 并行执行搜索
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(search_single_kb, knowledge_names_list))

        # 合并结果
        for result_set in results:
            all_results.extend(result_set)

        # 对结果进行重排序，按得分排序
        all_results = sorted(all_results, key=lambda x: x.get("score", float('inf')))

        # 只保留最好的top_k个结果
        return all_results[:self.default_top_k]

    def get_answer(self, user_id: str, query_text: str, knowledge_names: Optional[Union[str, List[str]]] = None) -> str:
        """
        综合检索与生成回答。

        Args:
            user_id: 用户ID
            query_text: 查询文本
            knowledge_names: 知识库名称或列表

        Returns:
            生成的回答
        """
        logger.info(f"Processing query for user: {user_id}. Query text: {query_text}")

        # 检查是否提供了多个知识库名称
        if isinstance(knowledge_names, str) and "," in knowledge_names:
            knowledge_names_list = [name.strip() for name in knowledge_names.split(",") if name.strip()]
            if len(knowledge_names_list) > 1:
                # 使用并行搜索优化多知识库查询
                contexts = self.parallel_search(user_id, query_text, knowledge_names_list)
            else:
                # 单个知识库，使用普通搜索
                contexts = self.search_similar_text(user_id, query_text, knowledge_names=knowledge_names)
        else:
            # 单个知识库或默认，使用普通搜索
            contexts = self.search_similar_text(user_id, query_text, knowledge_names=knowledge_names)

        # 如果没有找到上下文，返回错误信息
        if not contexts:
            logger.warning("No relevant context found in knowledge base.")
            raise HTTPException(status_code=404, detail="No relevant context found in knowledge base.")

        # 基于上下文和问题生成回答
        logger.info("Generating response based on retrieved context.")
        return self.generate_response(contexts, query_text)

    def answer_with_langchain_pipeline(self, user_id: str, query_text: str,
                                       knowledge_names: Optional[Union[str, List[str]]] = None) -> str:
        """
        使用LangChain管道来优化RAG流程

        Args:
            user_id: 用户ID
            query_text: 查询文本
            knowledge_names: 知识库名称或列表

        Returns:
            生成的回答
        """
        logger.info(f"Processing query with LangChain pipeline for user: {user_id}. Query: {query_text}")

        # 定义检索函数
        def retrieve_documents(query):
            return self.search_similar_text(user_id, query, knowledge_names=knowledge_names)

        # 定义格式化上下文函数
        def format_context(contexts):
            if not contexts:
                return "未找到相关信息。"

            context_texts = [f"来源: {ctx['source']}\n内容: {ctx['text']}" for ctx in contexts]
            return "\n\n---\n\n".join(context_texts)

        # 创建优化的提示模板
        rag_prompt = PromptTemplate.from_template("""
        你是一个专业的知识库助手。请根据下面提供的上下文信息，回答用户的问题。

        上下文信息:
        {context}

        用户问题: {question}

        请遵循以下规则：
        1. 仅基于提供的上下文信息回答问题，不要编造不存在的信息
        2. 如果上下文中没有足够的信息回答问题，请清楚地表明"基于提供的信息，我无法完全回答这个问题"，并提供上下文中可能相关的部分信息
        3. 如果上下文包含表格或结构化数据，请保持其格式清晰
        4. 提供具体、全面的回答，而不是过于笼统的信息
        5. 回答应该清晰、准确、有条理

        回答:
        """)

        # 构建RAG管道
        try:
            # 如果启用了HyDE，先生成假设性文档
            if self.hyde_enabled:
                logger.info("Using HyDE query transformation...")
                # 将原始查询转换为假设性文档
                hyde_query = self.generate_hypothetical_document(query_text)
                # 使用转换后的查询进行检索
                retrieval_query = hyde_query
                display_query = query_text  # 原始查询用于显示
            else:
                retrieval_query = query_text
                display_query = query_text

            # 构建并执行RAG管道
            rag_chain = (
                    {"context": RunnableLambda(lambda x: retrieve_documents(retrieval_query)) | RunnableLambda(
                        format_context),
                     "question": RunnablePassthrough()}
                    | rag_prompt
                    | self.chat_model
                    | StrOutputParser()
            )

            response = rag_chain.invoke(display_query)
            logger.info("Successfully generated response with LangChain pipeline.")
            return response

        except Exception as e:
            logger.error(f"Error in LangChain pipeline: {str(e)}")
            # 回退到常规方法
            logger.info("Falling back to standard RAG method...")
            return self.get_answer(user_id, query_text, knowledge_names)
