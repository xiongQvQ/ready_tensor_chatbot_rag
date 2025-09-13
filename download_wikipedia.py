

import wikipedia
import os

def download_wikipedia_article(topic, lang='zh'):
      """下载指定主题的维基百科文章"""
      wikipedia.set_lang(lang)

      try:
          # 搜索文章
          page = wikipedia.page(topic)

          # 创建文件名
          filename = f"data/{topic.replace(' ', '_')}.txt"

          # 保存文章内容
          with open(filename, 'w', encoding='utf-8') as f:
              f.write(f"标题: {page.title}\n")
              f.write(f"URL: {page.url}\n")
              f.write("=" * 50 + "\n\n")
              f.write(page.content)

          print(f"已下载: {filename}")
          return filename

      except wikipedia.exceptions.DisambiguationError as e:
          print(f"多个匹配项，请选择更具体的主题: {e.options[:5]}")
      except wikipedia.exceptions.PageError:
          print(f"未找到主题: {topic}")
      except Exception as e:
          print(f"下载失败: {e}")

# 使用示例
#download_wikipedia_article("Machine Learning")
download_wikipedia_article("python analysis", lang='en')

