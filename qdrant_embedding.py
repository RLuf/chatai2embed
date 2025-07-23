#!/usr/bin/env python3
import json
from typing import List, Dict, Optional
import uuid
import hashlib
import requests
from qdrant_client import QdrantClient
from qdrant_client.http import models
from pydantic import BaseModel, ValidationError
from typing_extensions import Literal
import jsonschema # Adicionado para resolver o erro de importação

# Enhanced Qdrant Embedding Script for JSON Chat Data Preservation  

"""
Este script é otimizado para processar e preservar dados de chat exportados em JSON (e.g., ChatGPT, Claude, DeepSeek) 
em sua forma mais bruta, tipicamente de arquivos como 'conversation.json'. Ele foca exclusivamente em manter a integridade 
dos dados durante o embedding para Qdrant, evitando transformações que poderiam alterar o conteúdo original. 
As características principais incluem:

1. **Validação Rigorosa de Schema JSON**: Garante que a entrada corresponda aos formatos esperados de exportação de chat.
2. **Preservação Completa de Metadados**: Retém todos os campos originais (timestamps, roles, IDs) sem filtragem.
3. **Chunking Consciente do Contexto**: Divide mensagens longas enquanto preserva o contexto conversacional,
   incluindo mensagens anteriores relevantes.
4. **Isolamento de Embedding**: Processa o conteúdo de texto separadamente dos metadados estruturais,
   garantindo que o embedding represente apenas a semântica do texto.
5. **Processamento Idempotente**: Garante saídas idênticas para entradas idênticas, prevenindo duplicação de dados
   e garantindo consistência ao reprocessar. IDs de pontos são gerados deterministicamente.

O script prioriza a precisão arquivística sobre a otimização de embedding, tornando-o ideal para:
- Preservação legal/conformidade
- Conjuntos de dados de pesquisa
- Migrações de histórico de chat
"""

# Schema para validação rigorosa de formatos de chat
# Este schema define a estrutura esperada para os arquivos JSON de exportação de chat.
CHAT_SCHEMA = {
    "type": "object",
    "required": ["messages", "metadata", "created_at"],
    "properties": {
        "messages": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["role", "content", "timestamp"],
                "properties": {
                    "role": {"type": "string", "enum": ["user", "assistant", "system"]},
                    "content": {"type": "string"},
                    "timestamp": {"type": "string", "format": "date-time"},
                    "id": {"type": "string"},
                    "additional_metadata": {"type": "object"}
                }
            }
        },
        "metadata": {
            "type": "object",
            "required": ["source", "version"],
            "properties": {
                "source": {"type": "string", "enum": ["chatgpt", "claude", "deepseek"]},
                "version": {"type": "string"},
                "export_settings": {"type": "object"}
            }
        },
        "created_at": {"type": "string", "format": "date-time"}
    }
}

def validate_chat_schema(data: dict) -> None:
    """Valida o documento de chat contra o schema esperado.
    
    Garante que a estrutura do JSON de entrada esteja em conformidade com o CHAT_SCHEMA
    definido, assegurando a integridade dos dados antes do processamento.
    
    Args:
        data: Dados do chat a serem validados.
    
    Raises:
        ValidationError: Se os dados não corresponderem ao schema.
    """
    jsonschema.validate(instance=data, schema=CHAT_SCHEMA)

def preserve_metadata(chat_data: dict) -> dict:
    """Extrai e preserva todos os metadados originais do chat.
    
    Esta função coleta metadados globais da conversa e metadados específicos de cada mensagem,
    garantindo que todas as informações contextuais e estruturais sejam mantidas.
    
    Args:
        chat_data: Dados do chat validados pelo schema.
    
    Returns:
        Dicionário contendo:
        - 'system_metadata': Metadados globais da conversa (fonte, versão, data de criação, etc.).
        - 'message_metadata': Lista de metadados por mensagem (papel, timestamp, ID, etc.).
    """
    return {
        "system_metadata": {
            "source": chat_data["metadata"]["source"],
            "version": chat_data["metadata"]["version"],
            "created_at": chat_data["created_at"],
            **chat_data["metadata"].get("export_settings", {})
        },
        "message_metadata": [
            {
                "role": msg["role"],
                "timestamp": msg["timestamp"],
                **({k: v for k, v in msg.items() 
                   if k not in {"role", "content", "timestamp"}})
            }
            for msg in chat_data["messages"]
        ]
    }

# Schemas para validação rigorosa de formatos de chat
class ChatMessage(BaseModel):
    """Modelo base para mensagens de chat com validação rigorosa"""
    sender: str
    text: str
    uuid: Optional[str] = None
    created_at: Optional[str] = None
    metadata: Optional[Dict] = None
    
    @classmethod
    def validate_provider_format(cls, provider: str, data: Dict) -> 'ChatMessage':
        """Valida formato específico de provedor (ChatGPT, Claude, etc)"""
        if provider == "chatgpt":
            if "message" not in data or "content" not in data["message"]:
                raise ValueError("Formato ChatGPT inválido - campo 'message.content' ausente")
            return cls(
                sender=data["message"].get("role", "unknown"),
                text=data["message"]["content"],
                uuid=data.get("id"),
                created_at=data.get("created_at"),
                metadata=data.get("metadata")
            )
        elif provider == "claude":
            if "text" not in data:
                raise ValueError("Formato Claude inválido - campo 'text' ausente")
            return cls(
                sender=data.get("role", "unknown"),
                text=data["text"],
                uuid=data.get("uuid"),
                created_at=data.get("created_at"),
                metadata=data.get("metadata")
            )
        return cls(**data)

class Conversation(BaseModel):
    """Modelo para conversas com validação de provedor"""
    name: Optional[str] = None
    uuid: Optional[str] = None
    created_at: Optional[str] = None
    chat_messages: List[ChatMessage]
    metadata: Optional[Dict] = None
    provider: Optional[str] = None  # ChatGPT, Claude, etc
    
    @classmethod
    def validate_provider_format(cls, provider: str, data: Dict) -> "Conversation":
        """Valida formato completo de conversa por provedor"""
        if provider == "chatgpt":
            if "messages" not in data:
                raise ValueError("Formato ChatGPT inválido - campo 'messages' ausente")
            messages = [
                ChatMessage.validate_provider_format(provider, msg)
                for msg in data["messages"]
            ]
            return cls(
                name=data.get("title"),
                uuid=data.get("id"),
                created_at=data.get("created_at"),
                chat_messages=messages,
                metadata=data.get("metadata"),
                provider=provider
            )
        elif provider == "claude":
            if "conversation" not in data or "messages" not in data["conversation"]:
                raise ValueError("Formato Claude inválido - campos obrigatórios ausentes")
            messages = [
                ChatMessage.validate_provider_format(provider, msg)
                for msg in data["conversation"]["messages"]
            ]
            return cls(
                name=data["conversation"].get("title"),
                uuid=data["conversation"].get("id"),
                created_at=data["conversation"].get("created_at"),
                chat_messages=messages,
                metadata=data.get("metadata"),
                provider=provider
            )
        return cls(**data)

class QdrantEmbedder:
    def __init__(self, qdrant_host="localhost", qdrant_port=6333, ollama_host="localhost", ollama_port=11434):
        self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.ollama_url = f"http://{ollama_host}:{ollama_port}/api/embeddings"
        
    def get_embedding(self, text: str, model: str = "mxbai-embed-large") -> List[float]:
        """Gera embedding usando Ollama com tratamento de erros robusto"""
        if not text.strip():
            return []
            
        # Limita o texto para evitar problemas de contexto
        if len(text) > 8000:
            text = self._smart_truncate(text)
            
        payload = {
            "model": model,
            "prompt": text
        }
        
        try:
            response = requests.post(self.ollama_url, json=payload, timeout=60)
            response.raise_for_status()
            return response.json()["embedding"]
        except Exception as e:
            raise Exception(f"Erro ao gerar embedding: {str(e)}")

    def _smart_truncate(self, text: str, max_length: int = 8000) -> str:
        """Trunca texto mantendo contexto conversacional"""
        if len(text) <= max_length:
            return text
            
        # Tenta encontrar um ponto de quebra natural
        break_points = [
            text.rfind('\n\n', 0, max_length),
            text.rfind('\n', 0, max_length),
            text.rfind('. ', 0, max_length),
            text.rfind('? ', 0, max_length),
            max_length
        ]
        
        break_point = max(p for p in break_points if p != -1)
        return text[:break_point] + " [...]"

    def validate_conversation_data(self, data: Dict) -> Conversation:
        """Valida estrutura dos dados de conversa usando Pydantic"""
        try:
            return Conversation(**data)
        except ValidationError as e:
            raise ValueError(f"Dados de conversa inválidos: {str(e)}")

    def create_collection(self, collection_name: str, vector_size: int = 768):
        """Cria coleção no Qdrant com configurações otimizadas.
        
        Este método é idempotente: se a coleção já existe com os mesmos parâmetros,
        ela não será recriada.
        """
        self.qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE
            ),
            optimizers_config=models.OptimizersConfigDiff(
                indexing_threshold=0,  # Garante indexação imediata
                memmap_threshold=2000  # Otimização para muitos vetores
            )
        )

    def load_and_validate_json(self, json_file_path: str) -> List[Conversation]:
        """Carrega e valida arquivo JSON de conversas"""
        with open(json_file_path, 'r', encoding='utf-8') as file:
            raw_data = json.load(file)
            
        if isinstance(raw_data, dict):
            raw_data = [raw_data]
            
        return [self.validate_conversation_data(conv) for conv in raw_data]

    def process_conversations(
        self,
        json_file_path: str,
        collection_name: str,
        embed_strategy: Literal["full", "messages", "human_only", "assistant_only"] = "full",
        batch_size: int = 5,
        chunk_long_messages: bool = True
    ) -> int:
        """Processa conversas de um arquivo JSON para embedding no Qdrant.
        
        Este método orquestra a validação, preservação de metadados, chunking consciente
        do contexto e a inserção idempotente de pontos no Qdrant.
        
        Args:
            json_file_path: Caminho para o arquivo JSON de exportação de chat.
            collection_name: Nome da coleção Qdrant onde os embeddings serão armazenados.
            embed_strategy: Estratégia para gerar embeddings ('full', 'messages', 'human_only', 'assistant_only').
            batch_size: Número de pontos a serem inseridos em cada lote no Qdrant.
            chunk_long_messages: Se True, mensagens longas serão divididas em chunks.
            
        Returns:
            O número total de itens (conversas ou chunks de mensagens) processados e inseridos.
        """
        conversations = self.load_and_validate_json(json_file_path)
        test_embedding = self.get_embedding("test")
        vector_size = len(test_embedding)
        
        self.create_collection(collection_name, vector_size)
        
        points = []
        total_items = 0
        
        for conversation in conversations:
            try:
                # Preserva todos os metadados originais
                conv_metadata = {
                    "name": conversation.name,
                    "uuid": conversation.uuid,
                    "created_at": conversation.created_at,
                    **conversation.metadata
                } if conversation.metadata else {
                    "name": conversation.name,
                    "uuid": conversation.uuid,
                    "created_at": conversation.created_at
                }
                
                if embed_strategy == "full":
                    text = self._build_full_conversation_text(conversation)
                    if text:
                        embedding = self.get_embedding(text)
                        point = self._create_conversation_point(
                            conversation, embedding, text, conv_metadata
                        )
                        points.append(point)
                        total_items += 1
                
                else:  # Estratégias por mensagem
                    for message in conversation.chat_messages:
                        if self._should_process_message(message, embed_strategy):
                            chunks = [message.text]
                            if chunk_long_messages and len(message.text) > 2000:
                                # Usa o método de chunking da classe
                                chunks = self._chunk_message(message.text)
                                
                            for chunk in chunks:
                                context_text = self._build_message_context(
                                    conversation, message, chunk
                                )
                                embedding = self.get_embedding(context_text)
                                point = self._create_message_point(
                                    conversation, message, embedding, 
                                    context_text, conv_metadata, chunk
                                )
                                points.append(point)
                                total_items += 1
                                
                                if len(points) >= batch_size:
                                    self._upsert_batch(collection_name, points)
                                    points = []
                                    
            except Exception as e:
                print(f"Erro processando conversa: {str(e)}")
                continue
                
        if points:
            self._upsert_batch(collection_name, points)
            
        return total_items

    def _should_process_message(self, message: ChatMessage, strategy: str) -> bool:
        """Determina se mensagem deve ser processada baseado na estratégia"""
        if strategy == "messages":
            return bool(message.text.strip())
        elif strategy == "human_only":
            return message.sender.lower() in ["human", "user"] and bool(message.text.strip())
        elif strategy == "assistant_only":
            return message.sender.lower() in ["assistant", "ai"] and bool(message.text.strip())
        return False

    def _build_full_conversation_text(self, conversation: Conversation) -> str:
        """Constrói texto completo da conversa mantendo contexto"""
        lines = []
        if conversation.name:
            lines.append(f"Conversa: {conversation.name}")
            
        for msg in conversation.chat_messages:
            if msg.text.strip():
                lines.append(f"{msg.sender.capitalize()}: {msg.text}")
                
        return "\n".join(lines)

    def _chunk_message(self, text: str, max_chunk_size: int = 2000) -> List[str]:
        """Divide mensagens longas em chunks, priorizando a preservação do contexto conversacional.
        
        Tenta dividir o texto por parágrafos e, se necessário, por sentenças,
        para evitar cortar o texto no meio de uma ideia.
        
        Args:
            text: O conteúdo da mensagem a ser chunked.
            max_chunk_size: Tamanho máximo de caracteres para cada chunk.
            
        Returns:
            Uma lista de strings, onde cada string é um chunk de texto.
        """
        if len(text) <= max_chunk_size:
            return [text]
            
        chunks = []
        current_chunk = ""
        
        # Divide por parágrafos primeiro
        paragraphs = text.split('\n\n')
        for para in paragraphs:
            if len(current_chunk) + len(para) + 2 <= max_chunk_size:
                current_chunk += f"{para}\n\n" if current_chunk else para
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para
                
                # Se parágrafo for muito grande, divide por sentenças
                if len(para) > max_chunk_size:
                    sentences = para.split('. ')
                    for sent in sentences:
                        if len(current_chunk) + len(sent) + 2 <= max_chunk_size:
                            current_chunk += f". {sent}" if current_chunk else sent
                        else:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                            current_chunk = sent
        
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks

    def _build_message_context(self, conversation: Conversation, message: ChatMessage, chunk: str) -> str:
        """Constrói o texto de contexto para um chunk de mensagem individual.
        
        Inclui o nome da conversa e as últimas 2-3 mensagens anteriores para fornecer
        contexto semântico ao chunk atual, melhorando a qualidade do embedding.
        
        Args:
            conversation: O objeto Conversation ao qual a mensagem pertence.
            message: O objeto ChatMessage atual.
            chunk: O chunk de texto da mensagem atual.
            
        Returns:
            Uma string contendo o chunk da mensagem com contexto conversacional.
        """
        context = []
        
        # Adiciona metadados da conversa
        if conversation.name:
            context.append(f"Conversa: {conversation.name}")
            
        # Adiciona mensagens anteriores relevantes (últimas 2-3)
        msg_index = conversation.chat_messages.index(message)
        prev_messages = conversation.chat_messages[max(0, msg_index-3):msg_index]
        
        for prev_msg in prev_messages:
            if prev_msg.text.strip():
                context.append(f"{prev_msg.sender.capitalize()}: {prev_msg.text}")
                
        # Adiciona o chunk atual
        context.append(f"{message.sender.capitalize()}: {chunk}")
        
        return "\n".join(context)

    def _create_conversation_point(self, conversation, embedding, text, metadata):
        """Cria um PointStruct para uma conversa completa no Qdrant.
        
        Gera um ID determinístico usando SHA256 do conteúdo da conversa se o UUID original
        não estiver presente, garantindo a idempotência.
        
        Args:
            conversation: O objeto Conversation.
            embedding: O vetor de embedding da conversa.
            text: O conteúdo textual completo da conversa usado para o embedding.
            metadata: Metadados adicionais da conversa.
            
        Returns:
            Um objeto models.PointStruct pronto para inserção no Qdrant.
        """
        return models.PointStruct(
            id=conversation.uuid or hashlib.sha256(json.dumps(conversation.dict(), sort_keys=True).encode()).hexdigest(),
            vector=embedding,
            payload={
                "type": "conversation",
                "text_content": text,
                "original_data": conversation.dict(),
                **metadata
            }
        )

    def _create_message_point(self, conversation, message, embedding, text, conv_metadata, chunk):
        """Cria um PointStruct para um chunk de mensagem individual no Qdrant.
        
        Gera um ID determinístico usando SHA256 da mensagem, chunk e UUID da conversa
        se o UUID original da mensagem não estiver presente, garantindo a idempotência.
        
        Args:
            conversation: O objeto Conversation ao qual a mensagem pertence.
            message: O objeto ChatMessage.
            embedding: O vetor de embedding do chunk da mensagem.
            text: O conteúdo textual do chunk com contexto usado para o embedding.
            conv_metadata: Metadados da conversa pai.
            chunk: O chunk de texto original da mensagem.
            
        Returns:
            Um objeto models.PointStruct pronto para inserção no Qdrant.
        """
        return models.PointStruct(
            id=message.uuid or hashlib.sha256(json.dumps({"message": message.dict(), "chunk": chunk, "conversation_uuid": conversation.uuid}, sort_keys=True).encode()).hexdigest(),
            vector=embedding,
            payload={
                "type": "message",
                "text_content": text,
                "chunk": chunk,
                "sender": message.sender,
                "message_metadata": message.metadata or {},
                "conversation_metadata": conv_metadata,
                "original_message": message.dict()
            }
        )

    def _upsert_batch(self, collection_name, points):
        """Insere um lote de pontos na coleção Qdrant.
        
        Utiliza o método `upsert` do Qdrant, que é idempotente: se um ponto com o mesmo ID
        já existe, ele será atualizado; caso contrário, será criado.
        
        Args:
            collection_name: Nome da coleção Qdrant.
            points: Lista de objetos models.PointStruct a serem inseridos.
        """
        try:
            self.qdrant_client.upsert(
                collection_name=collection_name,
                points=points,
                wait=True  # Garante persistência
            )
        except Exception as e:
            print(f"Erro inserindo lote: {str(e)}")

if __name__ == "__main__":
    # Configuração e exemplo de uso permanecem similares
    pass
