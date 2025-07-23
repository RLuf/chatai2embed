import React, { useState, useRef } from 'react';
import { Upload, Play, Settings, Database, MessageSquare, CheckCircle, XCircle, AlertCircle, Loader } from 'lucide-react';

const QdrantEmbeddingFrontend = () => {
  const [config, setConfig] = useState({
    qdrant_host: 'localhost',
    qdrant_port: 6333,
    ollama_host: 'localhost',
    ollama_port: 11434,
    embedding_model: 'mxbai-embed-large',
    collection_name: 'chat_embeddings',
    embed_strategy: 'full',
    batch_size: 5,
    chunk_long_messages: true,
    max_text_length: 8000,
    chunk_size: 2000
  });
  
  const [file, setFile] = useState(null);
  const [processing, setProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [logs, setLogs] = useState([]);
  const [connections, setConnections] = useState({
    qdrant: 'unknown',
    ollama: 'unknown'
  });
  const [activeTab, setActiveTab] = useState('config');
  
  const fileInputRef = useRef(null);

  const addLog = (message, type = 'info') => {
    const timestamp = new Date().toLocaleTimeString();
    setLogs(prev => [...prev, { timestamp, message, type }]);
  };

  const testConnections = async () => {
    addLog('Testando conexões...', 'info');
    
    // Testa Qdrant
    try {
      const qdrantResponse = await fetch(`http://${config.qdrant_host}:${config.qdrant_port}/collections`);
      if (qdrantResponse.ok) {
        setConnections(prev => ({ ...prev, qdrant: 'success' }));
        addLog('✅ Qdrant conectado com sucesso', 'success');
      } else {
        setConnections(prev => ({ ...prev, qdrant: 'error' }));
        addLog('❌ Erro ao conectar com Qdrant', 'error');
      }
    } catch (error) {
      setConnections(prev => ({ ...prev, qdrant: 'error' }));
      addLog(`❌ Qdrant não encontrado: ${error.message}`, 'error');
    }

    // Testa Ollama
    try {
      const ollamaResponse = await fetch(`http://${config.ollama_host}:${config.ollama_port}/api/tags`);
      if (ollamaResponse.ok) {
        setConnections(prev => ({ ...prev, ollama: 'success' }));
        addLog('✅ Ollama conectado com sucesso', 'success');
      } else {
        setConnections(prev => ({ ...prev, ollama: 'error' }));
        addLog('❌ Erro ao conectar com Ollama', 'error');
      }
    } catch (error) {
      setConnections(prev => ({ ...prev, ollama: 'error' }));
      addLog(`❌ Ollama não encontrado: ${error.message}`, 'error');
    }
  };

  const handleFileSelect = (event) => {
    const selectedFile = event.target.files[0];
    if (selectedFile) {
      if (!selectedFile.name.endsWith('.json')) {
        addLog('❌ Por favor, selecione um arquivo JSON válido', 'error');
        return;
      }
      setFile(selectedFile);
      addLog(`📁 Arquivo selecionado: ${selectedFile.name} (${(selectedFile.size / 1024 / 1024).toFixed(2)} MB)`, 'success');
    }
  };

  const generateEmbeddingScript = () => {
    if (!file) {
      addLog('❌ Selecione um arquivo JSON primeiro', 'error');
      return;
    }

    const script = `#!/usr/bin/env python3
import json
from typing import List, Dict, Optional
import uuid
import hashlib
import requests
from qdrant_client import QdrantClient
from qdrant_client.http import models
from pydantic import BaseModel, ValidationError
from typing_extensions import Literal
import jsonschema
import time
import logging
from tqdm import tqdm

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Schema para validação rigorosa de formatos de chat
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
    provider: Optional[str] = None
    
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
    def __init__(self, config):
        self.config = config
        self.qdrant_client = QdrantClient(host=config['qdrant_host'], port=config['qdrant_port'])
        self.ollama_url = f"http://{config['ollama_host']}:{config['ollama_port']}/api/embeddings"
        self.logger = logger
        
    def validate_connections(self):
        """Valida conexões com Qdrant e Ollama antes de processar"""
        try:
            # Testa Qdrant
            self.qdrant_client.get_collections()
            self.logger.info("✅ Conexão com Qdrant validada")
            
            # Testa Ollama
            response = requests.get(f"http://{self.config['ollama_host']}:{self.config['ollama_port']}/api/tags", timeout=10)
            response.raise_for_status()
            self.logger.info("✅ Conexão com Ollama validada")
            
            return True
        except Exception as e:
            raise ConnectionError(f"Erro de conectividade: {str(e)}")
        
    def get_embedding(self, text: str, model: str = None) -> List[float]:
        """Gera embedding usando Ollama com retry e backoff exponencial"""
        if not text.strip():
            return []
            
        model = model or self.config['embedding_model']
        
        if len(text) > self.config['max_text_length']:
            text = self._smart_truncate(text)
            
        payload = {
            "model": model,
            "prompt": text
        }
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(self.ollama_url, json=payload, timeout=120)
                response.raise_for_status()
                return response.json()["embedding"]
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    self.logger.warning(f"Timeout no embedding, tentativa {attempt + 1}/{max_retries}. Aguardando {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                raise Exception(f"Timeout após {max_retries} tentativas")
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    self.logger.warning(f"Erro na requisição, tentativa {attempt + 1}/{max_retries}. Aguardando {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                raise Exception(f"Erro na requisição: {str(e)}")

    def _smart_truncate(self, text: str, max_length: int = None) -> str:
        """Trunca texto mantendo contexto conversacional"""
        max_length = max_length or self.config['max_text_length']
        if len(text) <= max_length:
            return text
            
        break_points = [
            text.rfind('\\n\\n', 0, max_length),
            text.rfind('\\n', 0, max_length),
            text.rfind('. ', 0, max_length),
            text.rfind('? ', 0, max_length),
            max_length
        ]
        
        break_point = max(p for p in break_points if p != -1)
        return text[:break_point] + " [...]"

    def create_collection(self, collection_name: str, vector_size: int = 768):
        """Cria coleção no Qdrant com configurações otimizadas"""
        try:
            self.qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE
                ),
                optimizers_config=models.OptimizersConfigDiff(
                    indexing_threshold=0,
                    memmap_threshold=2000
                )
            )
            self.logger.info(f"✅ Coleção '{collection_name}' criada com sucesso")
        except Exception as e:
            if "already exists" in str(e).lower():
                self.logger.info(f"ℹ️ Coleção '{collection_name}' já existe")
            else:
                raise e

    def load_and_validate_json(self, json_file_path: str) -> List[Conversation]:
        """Carrega e valida arquivo JSON de conversas"""
        self.logger.info(f"📁 Carregando arquivo: {json_file_path}")
        
        with open(json_file_path, 'r', encoding='utf-8') as file:
            raw_data = json.load(file)
            
        if isinstance(raw_data, dict):
            raw_data = [raw_data]
        
        conversations = []
        for i, conv_data in enumerate(raw_data):
            try:
                conv = Conversation(**conv_data)
                conversations.append(conv)
            except ValidationError as e:
                self.logger.warning(f"⚠️ Conversa {i} inválida: {str(e)}")
                continue
                
        self.logger.info(f"✅ {len(conversations)} conversas válidas carregadas")
        return conversations

    def _chunk_message(self, text: str, max_chunk_size: int = None) -> List[str]:
        """Divide mensagens longas em chunks"""
        max_chunk_size = max_chunk_size or self.config['chunk_size']
        if len(text) <= max_chunk_size:
            return [text]
            
        chunks = []
        current_chunk = ""
        
        paragraphs = text.split('\\n\\n')
        for para in paragraphs:
            if len(current_chunk) + len(para) + 2 <= max_chunk_size:
                current_chunk += f"{para}\\n\\n" if current_chunk else para
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para
                
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

    def process_conversations_with_progress(
        self,
        json_file_path: str,
        collection_name: str,
        embed_strategy: str = "full",
        batch_size: int = 5,
        chunk_long_messages: bool = True
    ) -> int:
        """Processa conversas com barra de progresso"""
        
        # Valida conexões primeiro
        self.validate_connections()
        
        conversations = self.load_and_validate_json(json_file_path)
        
        # Teste de embedding para determinar dimensão do vetor
        self.logger.info("🧪 Testando geração de embedding...")
        test_embedding = self.get_embedding("test")
        vector_size = len(test_embedding)
        self.logger.info(f"📐 Dimensão do vetor: {vector_size}")
        
        self.create_collection(collection_name, vector_size)
        
        points = []
        total_items = 0
        
        with tqdm(total=len(conversations), desc="Processando conversas") as pbar:
            for conversation in conversations:
                try:
                    conv_metadata = {
                        "name": conversation.name,
                        "uuid": conversation.uuid,
                        "created_at": conversation.created_at,
                        **(conversation.metadata or {})
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
                                if chunk_long_messages and len(message.text) > self.config['chunk_size']:
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
                    self.logger.error(f"❌ Erro processando conversa: {str(e)}")
                    continue
                finally:
                    pbar.update(1)
                    
        if points:
            self._upsert_batch(collection_name, points)
            
        self.logger.info(f"🎉 Processamento concluído! {total_items} itens inseridos.")
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
                
        return "\\n".join(lines)

    def _build_message_context(self, conversation: Conversation, message: ChatMessage, chunk: str) -> str:
        """Constrói o texto de contexto para um chunk de mensagem individual"""
        context = []
        
        if conversation.name:
            context.append(f"Conversa: {conversation.name}")
            
        msg_index = conversation.chat_messages.index(message)
        prev_messages = conversation.chat_messages[max(0, msg_index-3):msg_index]
        
        for prev_msg in prev_messages:
            if prev_msg.text.strip():
                context.append(f"{prev_msg.sender.capitalize()}: {prev_msg.text}")
                
        context.append(f"{message.sender.capitalize()}: {chunk}")
        
        return "\\n".join(context)

    def _create_conversation_point(self, conversation, embedding, text, metadata):
        """Cria um PointStruct para uma conversa completa no Qdrant"""
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
        """Cria um PointStruct para um chunk de mensagem individual no Qdrant"""
        return models.PointStruct(
            id=message.uuid or hashlib.sha256(json.dumps({
                "message": message.dict(), 
                "chunk": chunk, 
                "conversation_uuid": conversation.uuid
            }, sort_keys=True).encode()).hexdigest(),
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
        """Insere um lote de pontos na coleção Qdrant"""
        try:
            self.qdrant_client.upsert(
                collection_name=collection_name,
                points=points,
                wait=True
            )
            self.logger.info(f"📝 Lote de {len(points)} pontos inserido com sucesso")
        except Exception as e:
            self.logger.error(f"❌ Erro inserindo lote: {str(e)}")

def main():
    # Configuração a partir do frontend
    config = ${JSON.stringify(config, null, 4)}
    
    # Inicialização
    embedder = QdrantEmbedder(config)
    
    # Processamento
    try:
        total_items = embedder.process_conversations_with_progress(
            json_file_path="${file.name}",  # Substitua pelo caminho real do arquivo
            collection_name=config['collection_name'],
            embed_strategy=config['embed_strategy'],
            batch_size=config['batch_size'],
            chunk_long_messages=config['chunk_long_messages']
        )
        
        print(f"\\n🎉 Processamento concluído com sucesso!")
        print(f"📊 Total de itens processados: {total_items}")
        print(f"🗃️ Coleção: {config['collection_name']}")
        
    except Exception as e:
        logger.error(f"❌ Erro durante o processamento: {str(e)}")
        raise

if __name__ == "__main__":
    main()
`;

    // Criar e baixar o arquivo
    const blob = new Blob([script], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'qdrant_embedding_enhanced.py';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    
    addLog('📥 Script Python gerado e baixado com sucesso!', 'success');
    addLog('💡 Execute: python qdrant_embedding_enhanced.py', 'info');
  };

  const ConnectionStatus = ({ status }) => {
    const icons = {
      success: <CheckCircle className="w-4 h-4 text-green-500" />,
      error: <XCircle className="w-4 h-4 text-red-500" />,
      unknown: <AlertCircle className="w-4 h-4 text-gray-400" />
    };
    return icons[status];
  };

  const LogEntry = ({ log }) => {
    const colors = {
      success: 'text-green-600',
      error: 'text-red-600',
      warning: 'text-yellow-600',
      info: 'text-blue-600'
    };
    
    return (
      <div className={`text-sm font-mono ${colors[log.type] || 'text-gray-600'}`}>
        <span className="text-gray-400">[{log.timestamp}]</span> {log.message}
      </div>
    );
  };

  return (
    <div className="max-w-6xl mx-auto p-6 bg-gray-50 min-h-screen">
      <div className="bg-white rounded-lg shadow-lg overflow-hidden">
        {/* Header */}
        <div className="bg-gradient-to-r from-blue-600 to-purple-600 text-white p-6">
          <h1 className="text-2xl font-bold flex items-center gap-2">
            <Database className="w-8 h-8" />
            Qdrant Embedding Frontend
          </h1>
          <p className="mt-2 opacity-90">Interface para processar embeddings de chat JSON no Qdrant</p>
        </div>

        {/* Tabs */}
        <div className="border-b border-gray-200">
          <nav className="flex">
            {[
              { id: 'config', label: 'Configuração', icon: Settings },
              { id: 'upload', label: 'Upload & Execução', icon: Upload },
              { id: 'logs', label: 'Logs', icon: MessageSquare }
            ].map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-2 px-6 py-3 font-medium border-b-2 transition-colors ${
                  activeTab === tab.id
                    ? 'border-blue-500 text-blue-600 bg-blue-50'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:bg-gray-50'
                }`}
              >
                <tab.icon className="w-4 h-4" />
                {tab.label}
              </button>
            ))}
          </nav>
        </div>

        {/* Content */}
        <div className="p-6">
          {activeTab === 'config' && (
            <div className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Conexões */}
                <div className="bg-gray-50 p-4 rounded-lg">
                  <h3 className="font-semibold mb-4 flex items-center gap-2">
                    <Database className="w-5 h-5" />
                    Configurações de Conexão
                  </h3>
                  
                  <div className="space-y-4">
                    <div>
                      <label className="block text-sm font-medium mb-1">Host Qdrant</label>
                      <input
                        type="text"
                        value={config.qdrant_host}
                        onChange={(e) => setConfig(prev => ({...prev, qdrant_host: e.target.value}))}
                        className="w-full p-2 border border-gray-300 rounded focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                      />
                    </div>
                    
                    <div>
                      <label className="block text-sm font-medium mb-1">Porta Qdrant</label>
                      <input
                        type="number"
                        value={config.qdrant_port}
                        onChange={(e) => setConfig(prev => ({...prev, qdrant_port: parseInt(e.target.value)}))}
                        className="w-full p-2 border border-gray-300 rounded focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                      />
                    </div>
                    
                    <div>
                      <label className="block text-sm font-medium mb-1">Host Ollama</label>
                      <input
                        type="text"
                        value={config.ollama_host}
                        onChange={(e) => setConfig(prev => ({...prev, ollama_host: e.target.value}))}
                        className="w-full p-2 border border-gray-300 rounded focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                      />
                    </div>
                    
                    <div>
                      <label className="block text-sm font-medium mb-1">Porta Ollama</label>
                      <input
                        type="number"
                        value={config.ollama_port}
                        onChange={(e) => setConfig(prev => ({...prev, ollama_port: parseInt(e.target.value)}))}
                        className="w-full p-2 border border-gray-300 rounded focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                      />
                    </div>
                  </div>
                  
                  <button
                    onClick={testConnections}
                    className="mt-4 w-full bg-blue-600 text-white py-2 px-4 rounded hover:bg-blue-700 transition-colors flex items-center justify-center gap-2"
                  >
                    <Database className="w-4 h-4" />
                    Testar Conexões
                  </button>
                  
                  <div className="mt-4 space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-sm">Qdrant:</span>
                      <ConnectionStatus status={connections.qdrant} />
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm">Ollama:</span>
                      <ConnectionStatus status={connections.ollama} />
                    </div>
                  </div>
                </div>

                {/* Configurações de Processamento */}
                <div className