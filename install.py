#!/usr/bin/env python3
import os
import json
import sys


def main():
    print("🚀 开始安装知识库插件...")

    if not os.path.exists('app.py'):
        print("❌ 错误：请在COW项目根目录下运行此脚本")
        sys.exit(1)

    print("📁 创建目录...")
    os.makedirs('plugins/knowledge_base', exist_ok=True)
    os.makedirs('chroma_db', exist_ok=True)

    print("📦 安装Python依赖...")
    os.system('pip3 install -q chromadb beautifulsoup4 readability-lxml requests openai')

    print("📝 创建插件文件...")

    # __init__.py
    with open('plugins/knowledge_base/__init__.py', 'w') as f:
        f.write('')

    # config.json
    config_data = {
        "enabled": True,
        "openai_api_key": "请填入你的OpenAI API Key",
        "chroma_persist_directory": "./chroma_db",
        "max_results": 5,
        "similarity_threshold": 0.7
    }
    with open('plugins/knowledge_base/config.json', 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=4, ensure_ascii=False)

    # vector_store.py
    vector_code = '''import chromadb
  import os
  import hashlib
  import logging

  logger = logging.getLogger(__name__)

  class VectorStore:
      def __init__(self, persist_directory="./chroma_db"):
          self.persist_directory = persist_directory
          os.makedirs(persist_directory, exist_ok=True)
          self.client = chromadb.PersistentClient(path=persist_directory)
          try:
              self.collection = self.client.get_collection("knowledge_base")
          except:
              self.collection = self.client.create_collection(name="knowledge_base")

      def add_document(self, content, metadata):
          try:
              doc_id = hashlib.md5(content.encode('utf-8')).hexdigest()
              existing = self.collection.get(ids=[doc_id])
              if existing['ids']:
                  return doc_id
              self.collection.add(documents=[content], metadatas=[metadata], ids=[doc_id])
              return doc_id
          except Exception as e:
              logger.error(f"添加文档失败: {e}")
              return None

      def search_similar(self, query, n_results=5):
          try:
              results = self.collection.query(query_texts=[query], n_results=n_results)
              formatted_results = []
              if results['documents'] and results['documents'][0]:
                  for i in range(len(results['documents'][0])):
                      formatted_results.append({
                          'content': results['documents'][0][i],
                          'metadata': results['metadatas'][0][i] if results['metadatas'][0] else {},
                          'distance': results['distances'][0][i] if results['distances'][0] else 0
                      })
              return formatted_results
          except Exception as e:
              logger.error(f"搜索失败: {e}")
              return []
  '''
    with open('plugins/knowledge_base/vector_store.py', 'w', encoding='utf-8') as f:
        f.write(vector_code)

    # article_parser.py
    article_code = '''import requests
  import re
  import logging
  from bs4 import BeautifulSoup
  from readability import Document

  logger = logging.getLogger(__name__)

  class ArticleParser:
      def __init__(self):
          self.headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

      def is_wechat_article(self, url):
          return 'mp.weixin.qq.com' in url

      def parse_article(self, url):
          if not self.is_wechat_article(url):
              return None
          try:
              response = requests.get(url, headers=self.headers, timeout=10)
              response.raise_for_status()
              response.encoding = response.apparent_encoding
              doc = Document(response.text)
              title = doc.title()
              content = doc.summary()
              soup = BeautifulSoup(content, 'html.parser')
              text_content = self._clean_text(soup.get_text())
              if len(text_content.strip()) < 100:
                  return None
              return {
                  'title': title,
                  'content': text_content,
                  'url': url,
                  'metadata': {'type': 'wechat_article', 'url': url, 'source': '微信公众号'}
              }
          except Exception as e:
              logger.error(f"解析文章失败: {e}")
              return None

      def _clean_text(self, text):
          text = re.sub(r'\\s+', ' ', text)
          text = re.sub(r'[^\\w\\s\\u4e00-\\u9fff，。！？；：""''（）【】《》、]', '', text)
          return text.strip()

      def chunk_content(self, content, chunk_size=1000, overlap=100):
          if len(content) <= chunk_size:
              return [content]
          chunks = []
          start = 0
          while start < len(content):
              end = start + chunk_size
              if end < len(content):
                  last_period = content.rfind('。', start, end)
                  if last_period > start:
                      end = last_period + 1
              chunk = content[start:end].strip()
              if chunk:
                  chunks.append(chunk)
              start = end - overlap if end < len(content) else end
          return chunks
  '''
    with open('plugins/knowledge_base/article_parser.py', 'w', encoding='utf-8') as f:
        f.write(article_code)

    # podcast_parser.py
    podcast_code = '''import requests
  import re
  import os
  import tempfile
  import logging
  import openai
  from bs4 import BeautifulSoup

  logger = logging.getLogger(__name__)

  class PodcastParser:
      def __init__(self, openai_api_key):
          self.openai_client = openai.OpenAI(api_key=openai_api_key)
          self.headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

      def is_xiaoyuzhou_link(self, url):
          return 'xiaoyuzhoufm.com' in url or 'xiaoyuzhou' in url.lower()

      def parse_podcast(self, url):
          if not self.is_xiaoyuzhou_link(url):
              return None
          try:
              podcast_info = self._get_podcast_info(url)
              if not podcast_info:
                  return None
              audio_file = self._download_audio(podcast_info['audio_url'])
              if not audio_file:
                  return None
              try:
                  transcript = self._transcribe_audio(audio_file)
                  if not transcript:
                      return None
                  return {
                      'title': podcast_info['title'],
                      'content': transcript,
                      'url': url,
                      'metadata': {'type': 'xiaoyuzhou_podcast', 'url': url, 'source': '小宇宙播客'}
                  }
              finally:
                  if os.path.exists(audio_file):
                      os.remove(audio_file)
          except Exception as e:
              logger.error(f"解析播客失败: {e}")
              return None

      def _get_podcast_info(self, url):
          try:
              response = requests.get(url, headers=self.headers, timeout=15)
              response.raise_for_status()
              soup = BeautifulSoup(response.text, 'html.parser')
              title_elem = soup.find('h1') or soup.find('title')
              title = title_elem.get_text().strip() if title_elem else "未知标题"
              audio_url = self._extract_audio_url(soup, url)
              if not audio_url:
                  return None
              return {'title': title, 'audio_url': audio_url}
          except Exception as e:
              logger.error(f"获取播客信息失败: {e}")
              return None

      def _extract_audio_url(self, soup, page_url):
          audio_tag = soup.find('audio')
          if audio_tag and audio_tag.get('src'):
              return audio_tag.get('src')
          scripts = soup.find_all('script')
          for script in scripts:
              if script.string:
                  matches = re.findall(r'https?://[^\\s"]+\\.(?:mp3|m4a|wav|aac)', script.string)
                  if matches:
                      return matches[0]
          return None

      def _download_audio(self, audio_url):
          try:
              temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
              temp_filename = temp_file.name
              temp_file.close()
              response = requests.get(audio_url, headers=self.headers, stream=True, timeout=30)
              response.raise_for_status()
              with open(temp_filename, 'wb') as f:
                  for chunk in response.iter_content(chunk_size=8192):
                      f.write(chunk)
              return temp_filename
          except Exception as e:
              logger.error(f"下载音频失败: {e}")
              return None

      def _transcribe_audio(self, audio_file):
          try:
              with open(audio_file, 'rb') as f:
                  transcript = self.openai_client.audio.transcriptions.create(
                      model="whisper-1", file=f, language="zh"
                  )
              return transcript.text
          except Exception as e:
              logger.error(f"音频转录失败: {e}")
              return None
  '''
    with open('plugins/knowledge_base/podcast_parser.py', 'w', encoding='utf-8') as f:
        f.write(podcast_code)

    # knowledge_base.py
    main_code = '''import os
  import re
  import logging
  from typing import Optional, List
  from bridge.context import Context, ContextType
  from bridge.reply import Reply, ReplyType
  from plugins import *
  from .vector_store import VectorStore
  from .article_parser import ArticleParser
  from .podcast_parser import PodcastParser

  logger = logging.getLogger(__name__)

  @plugins.register(
      name="KnowledgeBase",
      desire_priority=10,
      hidden=False,
      desc="智能知识库插件",
      version="1.0",
      author="Assistant"
  )
  class KnowledgeBasePlugin(Plugin):
      def __init__(self):
          super().__init__()
          try:
              config = super().load_config()
              if not config:
                  config = {
                      "enabled": True,
                      "openai_api_key": "",
                      "chroma_persist_directory": "./chroma_db",
                      "max_results": 5,
                      "similarity_threshold": 0.7
                  }
                  super().save_config(config)

              if not config.get("enabled", False):
                  return

              self.config = config
              persist_dir = config.get("chroma_persist_directory", "./chroma_db")
              self.vector_store = VectorStore(persist_dir)
              self.article_parser = ArticleParser()
              openai_key = config.get("openai_api_key")
              self.podcast_parser = PodcastParser(openai_key) if openai_key else None
              logger.info("知识库插件初始化成功")
          except Exception as e:
              logger.error(f"插件初始化失败: {e}")

      def on_handle_context(self, e_context):
          try:
              context = e_context['context']
              if context.type not in [ContextType.TEXT]:
                  return

              content = context.content.strip()
              urls = re.findall(r'https?://[^\\s\\u4e00-\\u9fff]+', content)

              if urls:
                  self._process_urls(urls, e_context)
              else:
                  self._search_knowledge_base(content, e_context)
          except Exception as e:
              logger.error(f"处理失败: {e}")

      def _process_urls(self, urls, e_context):
          processed_count = 0
          for url in urls:
              try:
                  if self.article_parser.is_wechat_article(url):
                      result = self.article_parser.parse_article(url)
                      if result:
                          chunks = self.article_parser.chunk_content(result['content'])
                          for chunk in chunks:
                              self.vector_store.add_document(chunk, result['metadata'])
                          processed_count += 1
                  elif self.podcast_parser and self.podcast_parser.is_xiaoyuzhou_link(url):
                      result = self.podcast_parser.parse_podcast(url)
                      if result:
                          self.vector_store.add_document(result['content'], result['metadata'])
                          processed_count += 1
              except Exception as e:
                  logger.error(f"处理URL失败: {e}")

          if processed_count > 0:
              reply = Reply(ReplyType.TEXT, f"✅ 已成功处理并保存 {processed_count} 个内容到知识库")
              e_context['reply'] = reply
              e_context.action = EventAction.BREAK_PASS

      def _search_knowledge_base(self, query, e_context):
          try:
              results = self.vector_store.search_similar(query, self.config.get("max_results", 5))
              if not results:
                  return

              threshold = self.config.get("similarity_threshold", 0.7)
              filtered_results = [r for r in results if r['distance'] < (1 - threshold)]
              if not filtered_results:
                  return

              context_parts = []
              for result in filtered_results[:3]:
                  content = result['content'][:300]
                  context_parts.append(f"参考内容：{content}")

              if context_parts:
                  enhanced_query = f"{query}\\n\\n{chr(10).join(context_parts)}"
                  e_context['context'].content = enhanced_query
          except Exception as e:
              logger.error(f"检索失败: {e}")
  '''
    with open('plugins/knowledge_base/knowledge_base.py', 'w', encoding='utf-8') as f:
        f.write(main_code)

    # 更新主配置
    main_config = {}
    if os.path.exists('plugins/plugins.json'):
        with open('plugins/plugins.json', 'r', encoding='utf-8') as f:
            main_config = json.load(f)

    main_config['knowledge_base'] = {'enabled': True, 'priority': 10}

    with open('plugins/plugins.json', 'w', encoding='utf-8') as f:
        json.dump(main_config, f, indent=2, ensure_ascii=False)

    print("🎉 知识库插件安装完成！")
    print("接下来请：")
    print("1. 配置API Key: nano plugins/knowledge_base/config.json")
    print("2. 重启项目: pkill -f app.py && nohup python3 app.py &")
    print("3. 测试功能: 发送微信文章或播客链接")


if __name__ == "__main__":
    main()
